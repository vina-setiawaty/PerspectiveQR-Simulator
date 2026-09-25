(function () {
  'use strict';

  // ----------------------------------------------------------------------
  // Geometry: a hip-roof solid.
  //   Base rectangle (L x W) at height 0, a ridge of length R (<= L) at
  //   height H, centered over the base in both directions.
  //   Three.js convention: Z = length (ridge direction), Y = height (up),
  //   X = width/depth -- rotated 90 deg from the "naive" X-length layout so
  //   that in top-down view (screen-up = -Z) the triangular hip ends land
  //   at the top/bottom of the frame and the trapezoids at the sides.
  //     A(W/2,0,-L/2)  B(W/2,0,L/2)  C(-W/2,0,L/2)  D(-W/2,0,-L/2)
  //     E(0,H,-R/2)    F(0,H,R/2)
  //   Faces: front A-B-F-E, back D-C-F-E, left hip A-D-E, right hip B-C-F.
  //   R=0 -> pyramid. R=L -> plain gable-roof prism (vertical end walls).
  // ----------------------------------------------------------------------

  const GRID_N = 28;        // mesh subdivisions per face (see plan: naive
                             // 4-corner UVs mis-locate ~2.5cm at the seam)
  const LIVE_BAKE_RES = 512;
  const PRINT_DPI = 300;
  const PRINT_PX_PER_CM = PRINT_DPI / 2.54; // final per-tile image res for the downloaded PDF
  const BUFFER_CM = 1;      // total QR buffer (0.5cm per edge)
  const BG_RGB = [244, 244, 240];
  const DARK_RGB = [17, 17, 17];
  const DEFAULT_FOV = 45;
  const TOP_VIEW_FOV = 60;
  const MIN_VIEW_DIST_CM = 0.5; // floor to avoid a divide-by-zero right at the ridge
  const MM_PER_CM = 10;
  const QR_EMBED_DEPTH_CM = 0.2;    // 2mm -- how deep the QR-module body reaches into the prism
  const QR_EMBED_PROTRUDE_CM = 0;   // flush with the surface, not raised above it -- no embossed look

  const state = {
    L: 21, W: 16, R: 10, H: 5,
    viewDist: 30, // scan/reveal distance above the ridge top -- see revealCameraHeight()
    qrText: 'https://interactivematerials.info/2025-11-coded-life',
    facesMode: 2,
    pageSize: 'A4',
    format: 'PDF', // 'PDF' | 'SVG' | 'PNG'
  };

  function fmtCm(n) { return `${Math.round(n * 10) / 10}cm`; }
  function fmtSlug(n) { return (Math.round(n * 10) / 10).toString().replace('.', 'p'); }
  function paramSlug(s) { return `L${fmtSlug(s.L)}-W${fmtSlug(s.W)}-R${fmtSlug(s.R)}-H${fmtSlug(s.H)}-D${fmtSlug(s.viewDist)}`; }
  function paramCaption(s) { return `L=${fmtCm(s.L)}  W=${fmtCm(s.W)}  R=${fmtCm(s.R)}  H=${fmtCm(s.H)}  scan=${fmtCm(s.viewDist)}`; }
  function paramCaptionShort(s) { return `L${fmtSlug(s.L)} W${fmtSlug(s.W)} R${fmtSlug(s.R)} H${fmtSlug(s.H)} D${fmtSlug(s.viewDist)}cm`; }

  let qrCache = { text: null, modules: null, size: 0, error: null };

  // ---------- geometry helpers ----------

  function computeVertices(s) {
    return {
      A: [ s.W / 2, 0, -s.L / 2],
      B: [ s.W / 2, 0,  s.L / 2],
      C: [-s.W / 2, 0,  s.L / 2],
      D: [-s.W / 2, 0, -s.L / 2],
      E: [0, s.H, -s.R / 2],
      F: [0, s.H,  s.R / 2],
    };
  }

  function faceCornerSets(v) {
    return {
      front: { p00: v.A, p10: v.B, p11: v.F, p01: v.E },
      back:  { p00: v.D, p10: v.C, p11: v.F, p01: v.E },
      left:  { p00: v.A, p10: v.D, p11: v.E, p01: v.E },
      right: { p00: v.B, p10: v.C, p11: v.F, p01: v.F },
      // Closes the bottom so the solid isn't hollow -- without it, orbiting
      // underneath looks straight into the interior and the backside of
      // the sloped faces' textures. Flat (all 4 corners at y=0), so it
      // needs no subdivision the way the sloped faces do.
      base:  { p00: v.A, p10: v.B, p11: v.C, p01: v.D },
    };
  }

  // 2D (cm-space), per-piece counterparts of faceCornerSets' 3D corners --
  // the same physical unfolding buildTrapezoidPiece/buildTrianglePiece use
  // for the flat SVG/PNG/PDF export, not yet placed within the shared
  // texture atlas (see buildAtlasLayout). bilinear(p00,p10,p11,p01,s,t) with
  // these reproduces invertFaceProjection's rawB/vPiece formulas exactly
  // (algebraically verified against them), so this is the same unfolding
  // wired into the live-preview path, not a new/different one.
  function pieceCornerSets2D(s) {
    const sh = Math.sqrt((s.W / 2) ** 2 + s.H ** 2);
    const sh2 = Math.sqrt(((s.L - s.R) / 2) ** 2 + s.H ** 2);
    return {
      sh, sh2,
      front: { p00: [-s.L / 2, 0], p10: [s.L / 2, 0], p11: [s.R / 2, sh], p01: [-s.R / 2, sh] },
      back:  { p00: [-s.L / 2, 2 * sh], p10: [s.L / 2, 2 * sh], p11: [s.R / 2, sh], p01: [-s.R / 2, sh] },
      left:  { p00: [-s.W / 2, 0], p10: [s.W / 2, 0], p11: [0, sh2], p01: [0, sh2] },
      right: { p00: [-s.W / 2, 0], p10: [s.W / 2, 0], p11: [0, sh2], p01: [0, sh2] },
    };
  }

  function bilinear2D(p00, p10, p11, p01, s, t) {
    const w00 = (1 - s) * (1 - t), w10 = s * (1 - t), w11 = s * t, w01 = (1 - s) * t;
    return [
      w00 * p00[0] + w10 * p10[0] + w11 * p11[0] + w01 * p01[0],
      w00 * p00[1] + w10 * p10[1] + w11 * p11[1] + w01 * p01[1],
    ];
  }

  // Where each piece sits within the shared atlas canvas -- the trapezoid
  // (front+back, joined at the ridge fold) along the top, and, when the hip
  // ends are shown, the two triangles centered side by side below it.
  function buildAtlasLayout(s, showHipEnds) {
    const pieceCorners = pieceCornerSets2D(s);
    const { sh, sh2 } = pieceCorners;
    const trapWidth = s.L, trapHeight = 2 * sh;
    const triWidth = s.W, triHeight = sh2;
    const widthCm = showHipEnds ? Math.max(trapWidth, 2 * triWidth) : trapWidth;
    const heightCm = trapHeight + (showHipEnds ? triHeight : 0);
    // Per face: where its own local (cm) origin lands in atlas cm-space.
    const centers = { front: [widthCm / 2, 0], back: [widthCm / 2, 0] };
    if (showHipEnds) {
      centers.left = [widthCm / 2 - triWidth / 2, trapHeight];
      centers.right = [widthCm / 2 + triWidth / 2, trapHeight];
    }
    return { widthCm, heightCm, pieceCorners, centers };
  }

  function atlasPositionCm(layout, faceKey, s, t) {
    const corners = layout.pieceCorners[faceKey];
    const [localX, localY] = bilinear2D(corners.p00, corners.p10, corners.p11, corners.p01, s, t);
    const [cx, baseY] = layout.centers[faceKey];
    return [localX + cx, localY + baseY];
  }

  // A face's mesh (s,t) grid point, placed within the shared atlas and
  // normalized to 0..1 -- the atlas equivalent of buildFaceGeometry's naive
  // `uvs[ui++] = s; uvs[ui++] = t`.
  function atlasUV(layout, faceKey, s, t) {
    const [x, y] = atlasPositionCm(layout, faceKey, s, t);
    return [x / layout.widthCm, y / layout.heightCm];
  }

  // P(s,t) = (1-s)(1-t)p00 + s(1-t)p10 + s t p11 + (1-s) t p01
  function bilinear(p00, p10, p11, p01, s, t, out) {
    const w00 = (1 - s) * (1 - t), w10 = s * (1 - t), w11 = s * t, w01 = (1 - s) * t;
    out[0] = w00 * p00[0] + w10 * p10[0] + w11 * p11[0] + w01 * p01[0];
    out[1] = w00 * p00[1] + w10 * p10[1] + w11 * p11[1] + w01 * p01[1];
    out[2] = w00 * p00[2] + w10 * p10[2] + w11 * p11[2] + w01 * p01[2];
    return out;
  }

  // `uvFn(s,t) => [u,v]` overrides the naive uv=(s,t) mapping -- used to
  // place a face within the shared texture atlas (see buildAtlasLayout)
  // instead of giving it the full [0,1] square to itself.
  function buildFaceGeometry(corners, n, uvFn) {
    const { p00, p10, p11, p01 } = corners;
    const positions = new Float32Array((n + 1) * (n + 1) * 3);
    const uvs = new Float32Array((n + 1) * (n + 1) * 2);
    const tmp = [0, 0, 0];
    let pi = 0, ui = 0;
    for (let j = 0; j <= n; j++) {
      const t = j / n;
      for (let i = 0; i <= n; i++) {
        const s = i / n;
        bilinear(p00, p10, p11, p01, s, t, tmp);
        positions[pi++] = tmp[0]; positions[pi++] = tmp[1]; positions[pi++] = tmp[2];
        if (uvFn) {
          const [uu, vv] = uvFn(s, t);
          uvs[ui++] = uu; uvs[ui++] = vv;
        } else {
          uvs[ui++] = s; uvs[ui++] = t;
        }
      }
    }
    const indices = [];
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const a = j * (n + 1) + i;
        const b = a + 1;
        const c = a + (n + 1);
        const d = c + 1;
        indices.push(a, b, c,  b, d, c);
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geo.setIndex(indices);
    return geo;
  }

  // ---------- QR ----------

  function ensureQR(text) {
    if (qrCache.text === text && (qrCache.modules || qrCache.error)) return qrCache;
    const input = text && text.trim().length ? text.trim() : ' ';
    try {
      const qr = QRCode.create(input, { errorCorrectionLevel: 'H' });
      qrCache = { text, modules: qr.modules, size: qr.modules.size, error: null };
    } catch (e) {
      qrCache = { text, modules: qrCache.modules || null, size: qrCache.size || 0, error: e && e.message ? e.message : 'Could not encode this text' };
    }
    return qrCache;
  }

  // The graphic is pre-warped for a real, finite viewing distance rather
  // than an idealized infinitely-far orthographic camera: a camera sitting
  // state.viewDist above the ridge top, looking straight down. Central
  // projection through that camera onto the y=0 base plane maps a point
  // (px,py,pz) to (px,pz)*camHeight/(camHeight-py) -- points higher up
  // (closer to the camera) are magnified more than the base, which is why
  // "seen from up there" the ridge appears wider than its true length.
  function effectiveViewDist(s) { return Math.max(s.viewDist, MIN_VIEW_DIST_CM); }

  function revealCameraHeight(s) { return s.H + effectiveViewDist(s); }

  // Apparent (as-seen-from-the-reveal-camera) length of the ridge -- the
  // width the two trapezoids pinch down to is R scaled by this factor, not
  // raw R, and it can end up *larger* than the base if the roof is tall
  // relative to the viewing distance.
  function apparentRidgeLength(s) {
    return s.R * revealCameraHeight(s) / effectiveViewDist(s);
  }

  function qrSizeCm(s) {
    let size;
    if (s.facesMode === 2) {
      const pinch = Math.min(s.L, apparentRidgeLength(s));
      size = Math.min(pinch, s.W) - BUFFER_CM;
    } else {
      size = Math.min(s.L, s.W) - BUFFER_CM;
    }
    return Math.max(0, size);
  }

  // ---------- live-preview texture baking (vector modules, per face) ----------
  //
  // Reuses the same closed-form inverse projection as the print export (see
  // invertFaceProjection further below): each dark module becomes a small
  // vector quadrilateral in this face's own (s,t) UV space, then Canvas2D's
  // natively antialiased path fill rasterizes it. No per-pixel point
  // sampling of the flat QR bitmap, so no aliasing at module edges
  // regardless of zoom level -- this replaced an earlier version that did
  // exactly that per-pixel sampling and was visibly jagged up close.
  function buildFaceModulePolygonsUV(corners, qr, sizeCm, camHeight) {
    const polys = [];
    if (!sizeCm || sizeCm <= 0) return polys;
    for (let row = 0; row < qr.size; row++) {
      for (let col = 0; col < qr.size; col++) {
        if (!qr.modules.get(row, col)) continue;
        const corners4 = moduleTargetCorners(qr, sizeCm, row, col);
        const cu = (corners4[0][0] + corners4[2][0]) / 2, cw = (corners4[0][1] + corners4[2][1]) / 2;
        const rc = invertFaceProjection(corners, cu, cw, camHeight);
        if (!rc || rc.s < -FACE_TOL || rc.s > 1 + FACE_TOL || rc.t < -FACE_TOL || rc.t > 1 + FACE_TOL) continue;
        const uvPts = corners4.map(([uu, ww]) => {
          const r = invertFaceProjection(corners, uu, ww, camHeight);
          if (!r) return null;
          return [Math.min(1, Math.max(0, r.s)), Math.min(1, Math.max(0, r.t))];
        });
        if (uvPts.some((p) => !p)) continue;
        polys.push(uvPts);
      }
    }
    return polys;
  }

  function bakeFaceTexture(corners, qr, sizeCm, res, camHeight) {
    const polys = buildFaceModulePolygonsUV(corners, qr, sizeCm, camHeight);
    const canvas = document.createElement('canvas');
    canvas.width = res; canvas.height = res;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = `rgb(${BG_RGB.join(',')})`;
    ctx.fillRect(0, 0, res, res);
    ctx.beginPath();
    polys.forEach((poly) => {
      poly.forEach(([s, t], i) => {
        const px = s * res, py = t * res;
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.closePath();
    });
    ctx.fillStyle = `rgb(${DARK_RGB.join(',')})`;
    ctx.fill();
    return canvas;
  }

  // ---------- Three.js scene ----------

  let scene, perspRenderer, topRenderer, perspCamera, topCamera, orbitControls;
  const meshes = {};

  function initScene() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xe9e9e4);

    ['front', 'back', 'left', 'right', 'base'].forEach((name) => {
      const geo = new THREE.BufferGeometry();
      const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });
      const mesh = new THREE.Mesh(geo, mat);
      scene.add(mesh);
      meshes[name] = mesh;
    });
    // Nudged a hair below y=0 so it isn't exactly coplanar with the sloped
    // faces' bottom edges -- otherwise the shared seam z-fights (flickering
    // moire) at grazing viewing angles. Imperceptible at real scene scale.
    meshes.base.position.y = -0.02;

    // No ground plane / grid -- the prism sits in open space so orbiting
    // all the way to directly overhead (or any other angle) is never
    // visually or functionally blocked by a "floor".
    const perspCanvas = document.getElementById('perspCanvas');
    perspRenderer = new THREE.WebGLRenderer({ canvas: perspCanvas, antialias: true });
    perspRenderer.outputEncoding = THREE.sRGBEncoding;
    perspCamera = new THREE.PerspectiveCamera(DEFAULT_FOV, 1, 1, 5000);
    // TrackballControls (arcball), not OrbitControls: OrbitControls
    // decomposes rotation into polar/azimuthal angles, and polar angle is
    // mathematically capped at [0, PI] no matter how it's configured --
    // dragging straight through the pole hits a hard stop there with
    // nowhere further to go. TrackballControls rotates the camera (and its
    // up vector) freely via arcball quaternions, so there's no pole and no
    // stopping point in any direction.
    orbitControls = new THREE.TrackballControls(perspCamera, perspCanvas);
    // staticMoving=true (no inertia/coasting): TrackballControls keeps
    // applying decaying leftover spin on every update() call after a drag
    // ends, with no public way to clear it -- which would otherwise nudge
    // the camera off-target the instant "View from Top" repositions it
    // right after the user was dragging. Dragging stops immediately on
    // mouse-up instead, trading coast-to-a-stop polish for the recenter
    // button always landing exactly where it should.
    orbitControls.staticMoving = true;
    orbitControls.minDistance = 10;
    orbitControls.maxDistance = 2000;

    // The top-down reveal pane is currently commented out of index.html
    // (not needed right now) -- everything below guards on its presence, so
    // uncommenting that <div> is the only thing needed to bring it back.
    const topCanvas = document.getElementById('topCanvas');
    if (topCanvas) {
      topRenderer = new THREE.WebGLRenderer({ canvas: topCanvas, antialias: true });
      topRenderer.outputEncoding = THREE.sRGBEncoding;
      // Also a real PerspectiveCamera: the "reveal" is now defined as the
      // view from a camera state.viewDist above the ridge top (see
      // revealCameraHeight), not an idealized orthographic one, so this
      // pane has to use the same finite-distance camera to match what was
      // actually baked into the textures.
      topCamera = new THREE.PerspectiveCamera(TOP_VIEW_FOV, 1, 1, 5000);
    }

    setDefaultPerspView();
    frameCameras();
    requestAnimationFrame(animate);
  }

  // Sets the initial 3/4 orbit framing. Called once at startup only --
  // rebuilding on a parameter change deliberately leaves the user's current
  // orbit position (or a "view from top" recenter) alone, same as any CAD
  // tool: editing a dimension shouldn't yank the camera back.
  function setDefaultPerspView() {
    const maxDim = Math.max(state.L, state.W, state.H * 2, 20);
    const dist = maxDim * 1.7;
    perspCamera.position.set(dist * 0.55, dist * 0.6, dist * 0.85);
    const focusY = state.H * 0.3;
    orbitControls.target.set(0, focusY, 0);
    perspCamera.lookAt(0, focusY, 0);
    orbitControls.update();
  }

  // Positions a camera exactly revealCameraHeight() above the base, centered
  // above the ridge, looking straight down, without ever touching
  // camera.up (that would desync OrbitControls' internal alignment
  // quaternion when applied to perspCamera). Optionally auto-fits the FOV
  // so the whole footprint stays in frame regardless of L/W.
  function positionRevealCamera(camera, autoFitFov) {
    const camY = revealCameraHeight(state);
    const m = new THREE.Matrix4().lookAt(
      new THREE.Vector3(0, camY, 0),
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, 0, -1)
    );
    camera.position.set(0, camY, 0);
    camera.quaternion.setFromRotationMatrix(m);
    if (autoFitFov) {
      const halfExtent = Math.max(state.L, state.W) / 2 + 5;
      const fovDeg = THREE.MathUtils.radToDeg(2 * Math.atan(halfExtent / camY));
      camera.fov = Math.min(Math.max(fovDeg, 10), 170);
      camera.updateProjectionMatrix();
    }
  }

  // "View from top" button: a one-shot recenter, like a CAD "look from
  // top" preset -- it moves the camera to the reveal position but leaves
  // the controls fully live, so the user can immediately keep
  // orbiting/zooming away from there. TrackballControls.update() always
  // ends with object.lookAt(target) using the *current* camera.up -- and
  // since free dragging rotates camera.up along with everything else (see
  // initScene), it can end up pointing anywhere, including parallel to
  // this dead-straight-down view direction, which would make lookAt
  // degenerate. Resetting up to a fixed, perpendicular value first avoids
  // that and also makes the button land on the same roll every time.
  function viewFromTop() {
    orbitControls.target.set(0, 0, 0);
    perspCamera.up.set(0, 0, -1);
    positionRevealCamera(perspCamera, true);
    orbitControls.update();
  }

  // Rebuilding on a parameter change deliberately leaves the user's current
  // orbit position (or a "view from top" recenter) alone, same as any CAD
  // tool: editing a dimension shouldn't yank the camera back. Only the
  // always-on reference pane (topCamera, no user controls) re-frames here.
  function frameCameras() {
    if (topCamera) positionRevealCamera(topCamera, true);
  }

  function resizeRendererToDisplaySize(renderer) {
    const canvas = renderer.domElement;
    const width = canvas.clientWidth, height = canvas.clientHeight;
    if (width === 0 || height === 0) return false;
    if (canvas.width !== width || canvas.height !== height) {
      renderer.setSize(width, height, false);
      return true;
    }
    return false;
  }

  function animate() {
    requestAnimationFrame(animate);
    orbitControls.update();

    if (resizeRendererToDisplaySize(perspRenderer)) {
      perspCamera.aspect = perspRenderer.domElement.clientWidth / perspRenderer.domElement.clientHeight;
      perspCamera.updateProjectionMatrix();
      orbitControls.handleResize(); // TrackballControls caches canvas bounds for drag math
    }
    if (topRenderer && resizeRendererToDisplaySize(topRenderer)) {
      topCamera.aspect = topRenderer.domElement.clientWidth / topRenderer.domElement.clientHeight;
      topCamera.updateProjectionMatrix();
    }

    perspRenderer.render(scene, perspCamera);
    if (topRenderer) topRenderer.render(scene, topCamera);
  }

  function disposeMap(material) {
    if (material.map) { material.map.dispose(); material.map = null; }
  }

  function rebuildAll() {
    const v = computeVertices(state);
    const fc = faceCornerSets(v);
    const qr = ensureQR(state.qrText);
    const sizeCm = qrSizeCm(state);
    const showHipEnds = state.facesMode === 4;
    const camHeight = revealCameraHeight(state);

    ['front', 'back'].forEach((name) => {
      const mesh = meshes[name];
      mesh.geometry.dispose();
      mesh.geometry = buildFaceGeometry(fc[name], GRID_N);
      const tex = new THREE.CanvasTexture(bakeFaceTexture(fc[name], qr, sizeCm, LIVE_BAKE_RES, camHeight));
      tex.flipY = false;
      tex.magFilter = THREE.LinearFilter;
      tex.minFilter = THREE.LinearMipmapLinearFilter;
      tex.generateMipmaps = true;
      // Without this, sharply foreshortened faces (the hip triangles read at
      // a far more oblique angle than the front-facing trapezoid slopes from
      // the default camera) get uniformly over-blurred: ordinary trilinear
      // filtering picks one mip level per pixel based on the WORST-case
      // (steepest) minification direction, so a face that's compressed hard
      // in only one screen direction still gets blurred in both.
      // Anisotropic filtering samples along the actual stretch direction
      // instead, keeping such faces legible.
      tex.anisotropy = perspRenderer.capabilities.getMaxAnisotropy();
      tex.needsUpdate = true;
      disposeMap(mesh.material);
      mesh.material.map = tex;
      mesh.material.color.set(0xffffff);
      mesh.material.needsUpdate = true;
    });

    ['left', 'right'].forEach((name) => {
      const mesh = meshes[name];
      mesh.geometry.dispose();
      mesh.geometry = buildFaceGeometry(fc[name], GRID_N);
      disposeMap(mesh.material);
      if (showHipEnds) {
        const tex = new THREE.CanvasTexture(bakeFaceTexture(fc[name], qr, sizeCm, LIVE_BAKE_RES, camHeight));
        tex.flipY = false;
        tex.magFilter = THREE.LinearFilter;
        tex.minFilter = THREE.LinearMipmapLinearFilter;
        tex.generateMipmaps = true;
        tex.anisotropy = perspRenderer.capabilities.getMaxAnisotropy();
        tex.needsUpdate = true;
        mesh.material.map = tex;
        mesh.material.color.set(0xffffff);
      } else {
        mesh.material.color.set(0xd8d8d2);
      }
      mesh.material.needsUpdate = true;
    });

    const baseMesh = meshes.base;
    baseMesh.geometry.dispose();
    baseMesh.geometry = buildFaceGeometry(fc.base, 1); // flat -- no subdivision needed
    disposeMap(baseMesh.material);
    baseMesh.material.color.set(0xd8d8d2);
    baseMesh.material.needsUpdate = true;

    frameCameras();
    updateReadouts(sizeCm, qr);
    updatePageCountReadout();
  }

  let rebuildScheduled = false;
  function scheduleRebuild() {
    if (rebuildScheduled) return;
    rebuildScheduled = true;
    requestAnimationFrame(() => {
      rebuildScheduled = false;
      rebuildAll();
    });
  }

  // ---------- readouts ----------

  function updateReadouts(sizeCm, qr) {
    const el = document.getElementById('qrSizeReadout');
    const warn = document.getElementById('warnReadout');
    el.textContent = state.facesMode === 2
      ? `QR graphic: ${sizeCm.toFixed(1)} cm square — fits the ridge as it appears from ${state.viewDist.toFixed(1)}cm above (minus 1cm buffer, capped by base width)`
      : `QR graphic: ${sizeCm.toFixed(1)} cm square — smaller of base L/W minus 1 cm buffer`;

    if (qr && qr.error) {
      warn.hidden = false;
      warn.textContent = 'Could not encode that text: ' + qr.error;
    } else if (sizeCm <= 0) {
      warn.hidden = false;
      warn.textContent = 'Ridge (or base) too small for a QR at this buffer — increase R, or L/W.';
    } else if (sizeCm < 5) {
      warn.hidden = false;
      warn.textContent = 'QR graphic is very small at this scale — likely not reliably scannable.';
    } else {
      warn.hidden = true;
    }
  }

  // ---------- flat, true-scale piece export (true vector) ----------
  //
  // Every dark QR module is drawn as an exact vector polygon on the flat
  // unfolded piece, instead of rasterizing the warp and hoping the pixel
  // grid is fine enough. That needs the *inverse* of the forward warp:
  // given a target point as it would appear to the reveal camera, find the
  // (s,t) on the face -- and, unusually for a perspective warp, this has a
  // closed form here, no iterative solve needed.
  //
  // Why: Y(s,t) = t*H for every face (bases at y=0, ridge at y=H always),
  // and for each face exactly one of X/Z is constant along s (the base
  // corners p00,p10 share it, and so do the ridge corners p11,p01) -- call
  // that one A(t), linear in t alone. The other, B(s,t), is linear in s for
  // any fixed t. Projecting gives target = A(t)*scale(t) for the first
  // coordinate, which is one linear equation in t (scale(t) is a Mobius
  // function of t, so this still resolves to a linear equation) -- solve
  // for t, then B(s,t) = otherTarget/scale(t) is known, and B is linear in
  // s so s falls out directly. See PLAN discussion for the full derivation.
  //
  // B(s,t), the raw (unprojected) value of that second coordinate, is also
  // *exactly* the flat unfolded piece's own local u-coordinate (proven the
  // same way the old raster bake's "u = X(s,t)" / "u = Z(s,t)" identity
  // was), so no separate conversion step is needed to place the solved
  // point on the print piece.

  function invertFaceProjection(corners, targetU, targetW, camHeight) {
    const { p00, p10, p11, p01 } = corners;
    const tAxis = Math.abs(p00[0] - p10[0]) < 1e-9 ? 0 : 2;
    const sAxis = tAxis === 0 ? 2 : 0;
    const A0 = p00[tAxis], A1 = p11[tAxis];
    const Y0 = p00[1], Y1 = p11[1];
    const targetA = tAxis === 0 ? targetU : targetW;
    const targetB = tAxis === 0 ? targetW : targetU;
    const D = camHeight;
    const denomT = D * (A1 - A0) + targetA * (Y1 - Y0);
    if (Math.abs(denomT) < 1e-9) return null;
    const t = (targetA * (D - Y0) - D * A0) / denomT;
    const Yt = Y0 + t * (Y1 - Y0);
    const scale = D / (D - Yt);
    const rawB = targetB / scale;
    const Bbase0 = p00[sAxis], Bbase1 = p10[sAxis], Btop0 = p01[sAxis], Btop1 = p11[sAxis];
    const Boffset = Bbase0 + t * (Btop0 - Bbase0);
    const Bwidth = (Bbase1 - Bbase0) + t * ((Btop1 - Btop0) - (Bbase1 - Bbase0));
    if (Math.abs(Bwidth) < 1e-9) return null;
    const s = (rawB - Boffset) / Bwidth;
    return { s, t, rawB };
  }

  // Forward counterpart of invertFaceProjection's projection model -- maps a
  // 3D point to its position in the shared "as seen from the reveal camera"
  // plane (the same (u,w) space moduleTargetCorners lays QR modules out in).
  // Used by the .3mf export to find each face's own boundary in that shared
  // plane, so a QR module can be clipped against it directly instead of
  // guessing which face it belongs to.
  function forwardProject(p, camHeight) {
    const scale = camHeight / (camHeight - p[1]);
    return [p[0] * scale, p[2] * scale];
  }

  function moduleTargetCorners(qr, sizeCm, row, col) {
    const modSize = sizeCm / qr.size;
    const half = sizeCm / 2;
    const u0 = -half + col * modSize, u1 = u0 + modSize;
    const w0 = -half + row * modSize, w1 = w0 + modSize;
    return [[u0, w0], [u1, w0], [u1, w1], [u0, w1]];
  }

  const FACE_TOL = 0.02; // slack on (s,t) validity, absorbs float error at face seams

  function buildTrapezoidPiece(s, qr, sizeCm, camHeight) {
    const sh = Math.sqrt((s.W / 2) ** 2 + s.H ** 2);
    const v = computeVertices(s);
    const fc = faceCornerSets(v);
    const modulePolys = [];
    for (let row = 0; row < qr.size; row++) {
      for (let col = 0; col < qr.size; col++) {
        if (!qr.modules.get(row, col)) continue;
        const corners4 = moduleTargetCorners(qr, sizeCm, row, col);
        const cu = (corners4[0][0] + corners4[2][0]) / 2, cw = (corners4[0][1] + corners4[2][1]) / 2;
        let faceKey = null;
        for (const key of ['front', 'back']) {
          const r = invertFaceProjection(fc[key], cu, cw, camHeight);
          if (r && r.s >= -FACE_TOL && r.s <= 1 + FACE_TOL && r.t >= -FACE_TOL && r.t <= 1 + FACE_TOL) { faceKey = key; break; }
        }
        if (!faceKey) continue;
        const piecePts = corners4.map(([uu, ww]) => {
          const r = invertFaceProjection(fc[faceKey], uu, ww, camHeight);
          if (!r) return null;
          const t = Math.min(1, Math.max(0, r.t));
          const vPiece = faceKey === 'front' ? t * sh : (2 * sh - t * sh);
          return [r.rawB, vPiece];
        });
        if (piecePts.some((p) => !p)) continue;
        modulePolys.push(piecePts);
      }
    }
    return {
      widthCm: s.L, heightCm: 2 * sh,
      outline: [[-s.L / 2, 0], [s.L / 2, 0], [s.R / 2, sh], [s.L / 2, 2 * sh], [-s.L / 2, 2 * sh], [-s.R / 2, sh]],
      foldLines: [sh],
      modulePolys,
    };
  }

  function buildTrianglePiece(s, side, qr, sizeCm, camHeight) {
    const sh2 = Math.sqrt(((s.L - s.R) / 2) ** 2 + s.H ** 2);
    const v = computeVertices(s);
    const fc = faceCornerSets(v);
    const faceKey = side;
    const modulePolys = [];
    for (let row = 0; row < qr.size; row++) {
      for (let col = 0; col < qr.size; col++) {
        if (!qr.modules.get(row, col)) continue;
        const corners4 = moduleTargetCorners(qr, sizeCm, row, col);
        const cu = (corners4[0][0] + corners4[2][0]) / 2, cw = (corners4[0][1] + corners4[2][1]) / 2;
        const rc = invertFaceProjection(fc[faceKey], cu, cw, camHeight);
        if (!rc || rc.s < -FACE_TOL || rc.s > 1 + FACE_TOL || rc.t < -FACE_TOL || rc.t > 1 + FACE_TOL) continue;
        const piecePts = corners4.map(([uu, ww]) => {
          const r = invertFaceProjection(fc[faceKey], uu, ww, camHeight);
          if (!r) return null;
          const t = Math.min(1, Math.max(0, r.t));
          return [r.rawB, t * sh2];
        });
        if (piecePts.some((p) => !p)) continue;
        modulePolys.push(piecePts);
      }
    }
    return { widthCm: s.W, heightCm: sh2, outline: [[-s.W / 2, 0], [s.W / 2, 0], [0, sh2]], foldLines: [], modulePolys };
  }

  // Rasterizes the vector piece (background fill of its true outline, dark
  // modules on top) at any target resolution -- Canvas2D's own path fill is
  // natively antialiased, so this needs no separate supersample/upscale
  // pass the way the old per-pixel raster bake did.
  function rasterizePiece(piece, pxPerCm) {
    const w = Math.max(1, Math.round(piece.widthCm * pxPerCm));
    const h = Math.max(1, Math.round(piece.heightCm * pxPerCm));
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    const toPx = ([x, y]) => [(x + piece.widthCm / 2) * pxPerCm, y * pxPerCm];
    ctx.beginPath();
    piece.outline.forEach((p, i) => {
      const [px, py] = toPx(p);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.fillStyle = `rgb(${BG_RGB.join(',')})`;
    ctx.fill();
    ctx.beginPath();
    piece.modulePolys.forEach((poly) => {
      poly.forEach((p, i) => {
        const [px, py] = toPx(p);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.closePath();
    });
    ctx.fillStyle = `rgb(${DARK_RGB.join(',')})`;
    ctx.fill();
    return canvas;
  }

  // ---------- print-ready, tiled PDF export ----------
  //
  // A piece is often bigger than any single sheet, so it's tiled across
  // multiple same-size pages at true 1:1 scale (like poster/banner tiling)
  // rather than shrunk to fit -- shrinking would defeat the entire point of
  // a "print at real size" export.

  const PAGE_SIZES = {
    A4: { w: 21.0, h: 29.7 },
    A3: { w: 29.7, h: 42.0 },
  };
  const PAGE_MARGIN_CM = 1.2; // usable-area margin on each edge of the sheet

  function computeTileGrid(pieceWcm, pieceHcm, usableW, usableH) {
    const nCols = Math.max(1, Math.ceil(pieceWcm / usableW));
    const nRows = Math.max(1, Math.ceil(pieceHcm / usableH));
    const tiles = [];
    for (let row = 0; row < nRows; row++) {
      for (let col = 0; col < nCols; col++) {
        const x0 = col * usableW;
        const y0 = row * usableH;
        tiles.push({
          row, col, nRows, nCols,
          x0, y0,
          w: Math.min(usableW, pieceWcm - x0),
          h: Math.min(usableH, pieceHcm - y0),
        });
      }
    }
    return { tiles, nCols, nRows };
  }

  const ITEM_GAP_CM = 0.6; // gap between distinct printable items sharing a page

  // Shelf packing (next-fit decreasing height): places items left-to-right
  // in rows, wrapping to a new shelf/page as needed. This is what lets two
  // triangles share one sheet, or land alongside trapezoid tiles that don't
  // fill the whole usable area, instead of every piece getting its own
  // dedicated page(s).
  function shelfPack(items, usableW, usableH, gapCm) {
    const sorted = items.slice().sort((a, b) => b.h - a.h);
    const pages = [];
    let page, shelfY, shelfH, cursorX;
    function newPage() { page = []; pages.push(page); shelfY = 0; shelfH = 0; cursorX = 0; }
    newPage();
    sorted.forEach((item) => {
      if (cursorX > 0 && cursorX + item.w > usableW + 1e-6) {
        cursorX = 0;
        shelfY += shelfH + gapCm;
        shelfH = 0;
      }
      if (shelfY + item.h > usableH + 1e-6) {
        newPage();
      }
      page.push({ item, x: cursorX, y: shelfY });
      cursorX += item.w + gapCm;
      shelfH = Math.max(shelfH, item.h);
    });
    return pages;
  }

  // Pure geometry: which pieces exist and how big they are, with no baking
  // -- cheap enough to call on every slider drag for the live page-count
  // readout, and reused as-is by the real export so the two always agree.
  function pieceDimensions(s) {
    const sh = Math.sqrt((s.W / 2) ** 2 + s.H ** 2);
    const dims = [{ id: 'trap', name: 'Trapezoid slopes (front+back)', w: s.L, h: 2 * sh }];
    if (s.facesMode === 4) {
      const sh2 = Math.sqrt(((s.L - s.R) / 2) ** 2 + s.H ** 2);
      dims.push({ id: 'left', name: 'Left hip triangle', w: s.W, h: sh2 });
      dims.push({ id: 'right', name: 'Right hip triangle', w: s.W, h: sh2 });
    }
    return dims;
  }

  // Orientation is chosen once for the whole document (from whichever piece
  // is largest by area, usually the trapezoid), then every piece -- tiled
  // if it's bigger than one sheet, a single item otherwise -- is packed
  // into that shared set of pages.
  function planLayout(s, pageKey) {
    const pieceDims = pieceDimensions(s);
    const mainPiece = pieceDims.reduce((a, b) => (a.w * a.h >= b.w * b.h ? a : b));
    const page = PAGE_SIZES[pageKey];
    const candidates = [
      { pageWcm: page.w, pageHcm: page.h },
      { pageWcm: page.h, pageHcm: page.w },
    ].map((o) => ({
      ...o,
      usableW: o.pageWcm - 2 * PAGE_MARGIN_CM,
      usableH: o.pageHcm - 2 * PAGE_MARGIN_CM,
    }));
    const chosen = candidates.reduce((best, o) => {
      const grid = computeTileGrid(mainPiece.w, mainPiece.h, o.usableW, o.usableH);
      const count = grid.nCols * grid.nRows;
      return !best || count < best.count ? { ...o, count } : best;
    }, null);

    const items = [];
    pieceDims.forEach(({ id, name, w, h }) => {
      if (w <= chosen.usableW && h <= chosen.usableH) {
        items.push({ id, name, w, h, x0: 0, y0: 0, row: 0, col: 0, nRows: 1, nCols: 1 });
      } else {
        const grid = computeTileGrid(w, h, chosen.usableW, chosen.usableH);
        grid.tiles.forEach((t) => items.push({
          id, name, w: t.w, h: t.h, x0: t.x0, y0: t.y0,
          row: t.row, col: t.col, nRows: t.nRows, nCols: t.nCols,
        }));
      }
    });

    const pages = shelfPack(items, chosen.usableW, chosen.usableH, ITEM_GAP_CM);
    return { pages, pageWcm: chosen.pageWcm, pageHcm: chosen.pageHcm, usableW: chosen.usableW, usableH: chosen.usableH };
  }

  function updatePageCountReadout() {
    const sizeCm = qrSizeCm(state);
    const el = document.getElementById('pageCountReadout');
    if (sizeCm <= 0) { el.textContent = ''; return; }
    const layout = planLayout(state, state.pageSize);
    const total = layout.pages.length;
    el.textContent = `≈ ${total} page${total === 1 ? '' : 's'} of ${state.pageSize}`;
  }

  function buildPageCanvas(pageItems, pieceCanvases, usableWcm, usableHcm) {
    const w = Math.max(1, Math.round(usableWcm * PRINT_PX_PER_CM));
    const h = Math.max(1, Math.round(usableHcm * PRINT_PX_PER_CM));
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);
    pageItems.forEach(({ item, x, y }) => {
      const { rasterCanvas, piece } = pieceCanvases[item.id];
      const srcPxPerCm = rasterCanvas.width / piece.widthCm;
      const dx = Math.round(x * PRINT_PX_PER_CM);
      const dy = Math.round(y * PRINT_PX_PER_CM);
      const dw = Math.round(item.w * PRINT_PX_PER_CM);
      const dh = Math.round(item.h * PRINT_PX_PER_CM);
      ctx.drawImage(
        rasterCanvas,
        item.x0 * srcPxPerCm, item.y0 * srcPxPerCm, item.w * srcPxPerCm, item.h * srcPxPerCm,
        dx, dy, dw, dh
      );
      ctx.strokeStyle = '#bbbbbb';
      ctx.lineWidth = Math.max(1, PRINT_PX_PER_CM * 0.015);
      ctx.strokeRect(dx, dy, dw, dh);
      ctx.fillStyle = '#555555';
      const fontPx = Math.round(PRINT_PX_PER_CM * 0.26);
      ctx.font = `${fontPx}px sans-serif`;
      const label = item.nRows === 1 && item.nCols === 1
        ? `${item.name} — print at 100% — ${paramCaptionShort(state)}`
        : `${item.name} — tile row ${item.row + 1}/${item.nRows}, col ${item.col + 1}/${item.nCols} — print at 100% — ${paramCaptionShort(state)}`;
      ctx.fillText(label, dx + fontPx * 0.4, dy + dh - fontPx * 0.5);
    });
    return canvas;
  }

  // Builds the vector geometry for every piece once (no rasterizing yet --
  // SVG export uses this directly; PNG/PDF rasterize it on demand below).
  function buildAllPieces() {
    const qr = ensureQR(state.qrText);
    const sizeCm = qrSizeCm(state);
    if (sizeCm <= 0) return null;
    const camHeight = revealCameraHeight(state);
    const result = {};
    result.trap = { name: 'Trapezoid slopes (front+back)', piece: buildTrapezoidPiece(state, qr, sizeCm, camHeight) };
    if (state.facesMode === 4) {
      result.left = { name: 'Left hip triangle', piece: buildTrianglePiece(state, 'left', qr, sizeCm, camHeight) };
      result.right = { name: 'Right hip triangle', piece: buildTrianglePiece(state, 'right', qr, sizeCm, camHeight) };
    }
    return result;
  }

  function pieceFileBase(id) { return id === 'trap' ? 'trapezoid-front-back' : `triangle-${id}`; }

  function downloadPdf() {
    const pieces = buildAllPieces();
    if (!pieces) return;
    const pieceCanvases = {};
    Object.entries(pieces).forEach(([id, { piece }]) => {
      pieceCanvases[id] = { piece, rasterCanvas: rasterizePiece(piece, PRINT_PX_PER_CM) };
    });
    const layout = planLayout(state, state.pageSize);
    const { jsPDF } = window.jspdf;
    let doc = null;
    layout.pages.forEach((pageItems) => {
      const pageCanvas = buildPageCanvas(pageItems, pieceCanvases, layout.usableW, layout.usableH);
      const dataUrl = pageCanvas.toDataURL('image/jpeg', 0.92);
      if (!doc) {
        doc = new jsPDF({ unit: 'cm', format: [layout.pageWcm, layout.pageHcm] });
      } else {
        doc.addPage([layout.pageWcm, layout.pageHcm]);
      }
      doc.addImage(dataUrl, 'JPEG', PAGE_MARGIN_CM, PAGE_MARGIN_CM, layout.usableW, layout.usableH);
    });
    doc.save(`qr-hip-roof_${paramSlug(state)}_${state.pageSize}.pdf`);
  }

  // ---------- standalone SVG / PNG export (one file per piece, true scale) ----------

  const CAPTION_HEIGHT_CM = 1.4; // extra strip below the shape for name + parameters

  function rgbToHex([r, g, b]) { return '#' + [r, g, b].map((v) => v.toString(16).padStart(2, '0')).join(''); }
  function escapeXml(str) { return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

  // SVG export: the modules are already exact vector polygons, so this is
  // just plain vector shapes -- no embedded raster image, no DPI ceiling.
  function svgFromVectorPiece(name, piece) {
    const { widthCm, heightCm, outline, foldLines, modulePolys } = piece;
    const totalHeightCm = heightCm + CAPTION_HEIGHT_CM;
    const outlinePts = outline.map(([x, y]) => `${x.toFixed(3)},${y.toFixed(3)}`).join(' ');
    const moduleD = modulePolys.map((poly) =>
      'M' + poly.map(([x, y]) => `${x.toFixed(3)},${y.toFixed(3)}`).join('L') + 'Z'
    ).join('');
    const folds = foldLines.map((y) =>
      `<line x1="${(-widthCm / 2).toFixed(3)}" y1="${y.toFixed(3)}" x2="${(widthCm / 2).toFixed(3)}" y2="${y.toFixed(3)}" stroke="#999999" stroke-width="0.05" stroke-dasharray="0.3,0.3"/>`
    ).join('\n  ');
    const fs1 = 0.42, fs2 = 0.36;
    return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${widthCm}cm" height="${totalHeightCm}cm" viewBox="${-widthCm / 2} 0 ${widthCm} ${totalHeightCm}">
  <polygon points="${outlinePts}" fill="${rgbToHex(BG_RGB)}"/>
  <path d="${moduleD}" fill="${rgbToHex(DARK_RGB)}"/>
  <polygon points="${outlinePts}" fill="none" stroke="#000000" stroke-width="0.06"/>
  ${folds}
  <text x="${(-widthCm / 2 + 0.3).toFixed(3)}" y="${(heightCm + CAPTION_HEIGHT_CM * 0.55).toFixed(3)}" font-size="${fs1}" font-family="sans-serif" fill="#333333">${escapeXml(name)}</text>
  <text x="${(-widthCm / 2 + 0.3).toFixed(3)}" y="${(heightCm + CAPTION_HEIGHT_CM * 0.9).toFixed(3)}" font-size="${fs2}" font-family="sans-serif" fill="#333333">${escapeXml(paramCaption(state))}</text>
</svg>`;
  }

  // PNG export reuses the same rasterizer as the PDF path, then appends a
  // plain caption strip (this one has to be baked in, since PNG has no text
  // layer of its own).
  function withCaptionStrip(source, widthCm, captionLine1, captionLine2) {
    const pxPerCm = source.width / widthCm;
    const capPx = Math.round(CAPTION_HEIGHT_CM * pxPerCm);
    const canvas = document.createElement('canvas');
    canvas.width = source.width;
    canvas.height = source.height + capPx;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(source, 0, 0);
    ctx.fillStyle = '#333333';
    const fontPx = Math.round(pxPerCm * 0.32);
    ctx.font = `${fontPx}px sans-serif`;
    ctx.fillText(captionLine1, pxPerCm * 0.3, source.height + capPx * 0.48);
    if (captionLine2) ctx.fillText(captionLine2, pxPerCm * 0.3, source.height + capPx * 0.88);
    return canvas;
  }

  function downloadBlob(filename, content, type) {
    const blob = content instanceof Blob ? content : new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  }

  function dataUrlToBlob(dataUrl) {
    const [meta, b64] = dataUrl.split(',');
    const mime = meta.match(/:(.*?);/)[1];
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return new Blob([arr], { type: mime });
  }

  function downloadSvg() {
    const pieces = buildAllPieces();
    if (!pieces) return;
    Object.entries(pieces).forEach(([id, { name, piece }], i) => {
      const svg = svgFromVectorPiece(name, piece);
      setTimeout(() => downloadBlob(`${pieceFileBase(id)}_${paramSlug(state)}.svg`, svg, 'image/svg+xml'), i * 350);
    });
  }

  function downloadPng() {
    const pieces = buildAllPieces();
    if (!pieces) return;
    Object.entries(pieces).forEach(([id, { name, piece }], i) => {
      const raster = rasterizePiece(piece, PRINT_PX_PER_CM);
      const composed = withCaptionStrip(raster, piece.widthCm, name, paramCaption(state));
      setTimeout(() => downloadBlob(
        `${pieceFileBase(id)}_${paramSlug(state)}.png`,
        dataUrlToBlob(composed.toDataURL('image/png')),
        'image/png'
      ), i * 350);
    });
  }

  // ---------- 3D model (.3mf) export for multi-color 3D printing ----------
  //
  // Two printable parts sharing one object, so a multi-material slicer
  // (PrusaSlicer/OrcaSlicer/Bambu Studio) loads them pre-assembled and ready
  // for a per-part filament/color assignment:
  //   - "Prism body": the unmodified hip-roof solid, full depth, one color.
  //   - "QR embed body": one small box per dark module, sitting flush with
  //     the outer surface and reaching QR_EMBED_DEPTH_CM into the solid.
  // The QR boxes deliberately overlap the (un-pocketed) prism volume rather
  // than being boolean-subtracted from it -- these slicers resolve that
  // themselves at slice time (a part later in the list wins in the region
  // where parts overlap, the standard technique for embedding a logo/pattern
  // in a second color), so no CSG engine is needed here.

  function vSub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function vAdd(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
  function vScale(a, k) { return [a[0] * k, a[1] * k, a[2] * k]; }
  function vCross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  function vDot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function vLen(a) { return Math.sqrt(vDot(a, a)); }
  function vNormalize(a) { const l = vLen(a); return l < 1e-12 ? [0, 0, 0] : [a[0] / l, a[1] / l, a[2] / l]; }

  // Appends one triangle's 3 vertices (9 floats) to `out`, choosing winding
  // order so its normal faces away from `interiorRef` -- a point known to
  // sit inside the solid. This sidesteps hand-tracking winding per face.
  // Zero-area triangles are dropped rather than emitted: the hip-end faces
  // are built from a "collapsed quad" (two corners coincide, see
  // faceCornerSets) that fans out to a point, and the last row of its grid
  // is degenerate by construction -- left in, that's exactly the kind of
  // defect a slicer's mesh-repair step flags on import.
  function pushTriangle(out, interiorRef, p0, p1, p2) {
    const n = vCross(vSub(p1, p0), vSub(p2, p0));
    if (vLen(n) < 1e-9) return;
    const ordered = vDot(n, vSub(p0, interiorRef)) >= 0 ? [p0, p1, p2] : [p0, p2, p1];
    ordered.forEach((p) => out.push(p[0], p[1], p[2]));
  }

  // A single representative outward normal for a whole (possibly gently
  // curved) face, from its 4 corners rather than a per-module quad. Every
  // module box on a face offsets along this SAME vector, so touching
  // modules' shared edges land on bit-identical 3D points -- required for
  // buildQrEmbedTriangles' neighbor-adjacency wall-skipping below to
  // actually produce coincident (weldable) geometry instead of a sliver
  // gap or overlap between boxes whose normals would otherwise differ by
  // the surface's curvature.
  function computeFaceOutwardNormal(corners, interiorRef) {
    const { p00, p10, p11, p01 } = corners;
    const n = vNormalize(vCross(vSub(p10, p00), vSub(p01, p00)));
    const centroid = vScale(vAdd(vAdd(p00, p10), vAdd(p11, p01)), 0.25);
    return vDot(n, vSub(centroid, interiorRef)) >= 0 ? n : vScale(n, -1);
  }

  // Reads a THREE.BufferGeometry's indexed triangles into the flat,
  // consistently-wound export list.
  function appendGeometryTriangles(geo, interiorRef, out) {
    const pos = geo.attributes.position.array;
    const idx = geo.index.array;
    for (let i = 0; i < idx.length; i += 3) {
      const i0 = idx[i] * 3, i1 = idx[i + 1] * 3, i2 = idx[i + 2] * 3;
      pushTriangle(
        out, interiorRef,
        [pos[i0], pos[i0 + 1], pos[i0 + 2]],
        [pos[i1], pos[i1 + 1], pos[i1 + 2]],
        [pos[i2], pos[i2 + 1], pos[i2 + 2]]
      );
    }
  }

  // The unmodified hip-roof solid (front/back/left/right slopes + base),
  // watertight, at full depth -- same shape as the live preview, just
  // exported as flat-wound triangles instead of a shaded THREE mesh.
  function buildPrismSolidTriangles(s) {
    const v = computeVertices(s);
    const fc = faceCornerSets(v);
    const interiorRef = [0, s.H * 0.3, 0]; // same interior point used to frame the perspective camera -- safely inside the solid for any valid L/W/R/H
    const out = [];
    ['front', 'back', 'left', 'right'].forEach((key) => {
      const geo = buildFaceGeometry(fc[key], GRID_N);
      appendGeometryTriangles(geo, interiorRef, out);
      geo.dispose();
    });
    // Subdivided at the same GRID_N as the sloped faces (even though flat
    // and needing none of its own) so its boundary lands on the exact same
    // points as their bottom edges -- a coarser base grid would leave a
    // T-junction seam there (many fine edge-segments on the sloped side
    // with no matching edge to weld against on the base side).
    const baseGeo = buildFaceGeometry(fc.base, GRID_N);
    appendGeometryTriangles(baseGeo, interiorRef, out);
    baseGeo.dispose();
    return out;
  }

  // Each dark QR module is a small square in the shared "as seen from the
  // reveal camera" plane (moduleTargetCorners) -- the same plane
  // forwardProject maps 3D points into. A face's own boundary, projected
  // into that same plane via forwardProject on its corners, is therefore
  // directly comparable to a module's square in that plane: clipping the
  // module against the face boundary (Sutherland-Hodgman -- both are
  // convex, so the result is always a single convex polygon) gives exactly
  // the portion of that module which belongs on this face, and nothing
  // else.
  //
  // This replaces an earlier approach that picked, per module CORNER, a
  // single "best-fit" face and forced that corner onto it: a module
  // straddling a seam between two differently-angled faces (front vs. a hip
  // triangle, especially in 4-slope mode) would then get corners offset
  // along two unrelated face normals, warping its box into a non-planar
  // shape -- the root cause of the QR embed body coming out broken/non-
  // manifold on 4 slopes. Clipping instead gives each sub-polygon to
  // exactly one face, so every extruded box is flat and built from a
  // single normal, with no per-box orientation guesswork needed (see
  // buildExtrudedPrism) and no whole-mesh seam-repair pass required
  // afterwards: two faces sharing an edge clip against the same shared line
  // in this plane, so their sub-polygons land on identical 3D points once
  // each is mapped back through its own (different) face parametrization.

  function polygonSignedArea(poly) {
    let a = 0;
    for (let i = 0; i < poly.length; i++) {
      const [x1, y1] = poly[i], [x2, y2] = poly[(i + 1) % poly.length];
      a += x1 * y2 - x2 * y1;
    }
    return a / 2;
  }

  function cross2(a, b, p) { return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]); }

  function lineIntersect2D(a1, a2, b1, b2) {
    const A1 = a2[1] - a1[1], B1 = a1[0] - a2[0], C1 = A1 * a1[0] + B1 * a1[1];
    const A2 = b2[1] - b1[1], B2 = b1[0] - b2[0], C2 = A2 * b1[0] + B2 * b1[1];
    const det = A1 * B2 - A2 * B1;
    if (Math.abs(det) < 1e-12) return b2;
    return [(B2 * C1 - B1 * C2) / det, (A1 * C2 - A2 * C1) / det];
  }

  // Sutherland-Hodgman clipping of `subject` against convex polygon `clip`
  // -- works for either winding order of `clip` (its own signed area picks
  // the "inside" sign), which matters here since a face boundary's winding
  // after forwardProject depends on that face's own corner order.
  //
  // Each output point is tagged `boundary: true` iff it was newly created by
  // an intersection (i.e. it lies exactly ON the clip edge) rather than kept
  // from the original subject polygon -- buildQrEmbedTriangles uses that tag
  // to tell a module's true interior corners from the seam points where it
  // was actually cut, since only the latter need a blended offset normal
  // (see blendedNormalAt).
  function clipPolygonConvex(subject, clip) {
    let output = subject.map((p) => ({ pt: p, boundary: false }));
    const sign = polygonSignedArea(clip) >= 0 ? 1 : -1;
    for (let i = 0; i < clip.length && output.length; i++) {
      const c1 = clip[i], c2 = clip[(i + 1) % clip.length];
      if (Math.abs(c1[0] - c2[0]) < 1e-9 && Math.abs(c1[1] - c2[1]) < 1e-9) continue; // degenerate edge (triangular face's collapsed corner)
      const input = output;
      output = [];
      for (let j = 0; j < input.length; j++) {
        const curr = input[j], prev = input[(j - 1 + input.length) % input.length];
        const currIn = sign * cross2(c1, c2, curr.pt) >= -1e-9;
        const prevIn = sign * cross2(c1, c2, prev.pt) >= -1e-9;
        if (currIn) {
          if (!prevIn) output.push({ pt: lineIntersect2D(c1, c2, prev.pt, curr.pt), boundary: true });
          output.push(curr);
        } else if (prevIn) {
          output.push({ pt: lineIntersect2D(c1, c2, prev.pt, curr.pt), boundary: true });
        }
      }
    }
    return output;
  }

  // Whether (u,w) lies on the perimeter of convex polygon `poly` (within
  // `tol`) -- point-to-segment distance against every edge.
  function pointOnPolygonBoundary(poly, u, w, tol) {
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i], b = poly[(i + 1) % poly.length];
      const abx = b[0] - a[0], aby = b[1] - a[1];
      const len2 = abx * abx + aby * aby;
      if (len2 < 1e-12) continue;
      const t = Math.max(0, Math.min(1, ((u - a[0]) * abx + (w - a[1]) * aby) / len2));
      const dx = u - (a[0] + t * abx), dy = w - (a[1] + t * aby);
      if (dx * dx + dy * dy < tol * tol) return true;
    }
    return false;
  }

  // The offset normal for a clip-created (seam) vertex: a module cut between
  // two faces has its two pieces meet only along the shared top edge, since
  // each would otherwise offset that same edge into the solid along its OWN
  // face's normal, leaving two DIFFERENT bottom edges (a non-manifold edge
  // where they meet, 4 faces converging on one line instead of 2). Using the
  // faces' averaged normal instead, for BOTH pieces, makes them offset that
  // shared edge identically -- their walls there become fully coincident and
  // get deduplicated by resolveWallCandidates, same as two same-face
  // neighbors. A point that isn't a genuine inter-face seam (e.g. the outer
  // bottom rim, which borders no other QR-textured face) falls back to the
  // single owning face's own normal, unchanged.
  function blendedNormalAt(faceKeys, faceBoundaries, faceNormals, ownKey, u, w) {
    let sum = faceNormals[ownKey], blended = false;
    faceKeys.forEach((otherKey) => {
      if (otherKey === ownKey) return;
      if (pointOnPolygonBoundary(faceBoundaries[otherKey], u, w, 1e-4)) {
        sum = vAdd(sum, faceNormals[otherKey]);
        blended = true;
      }
    });
    return blended ? vNormalize(sum) : faceNormals[ownKey];
  }

  // A face's own boundary, in the same shared plane moduleTargetCorners
  // lays QR modules out in -- straight 3D edges project to straight lines
  // under this central projection, so this is just forwardProject on the
  // face's corners (deduplicated, since a hip triangle's corners come in as
  // a collapsed quad -- see faceCornerSets).
  function faceBoundaryUW(corners, camHeight) {
    const pts = [corners.p00, corners.p10, corners.p11, corners.p01].map((p) => forwardProject(p, camHeight));
    return pts.filter((p, i) => {
      const prev = pts[(i - 1 + pts.length) % pts.length];
      return Math.abs(p[0] - prev[0]) > 1e-9 || Math.abs(p[1] - prev[1]) > 1e-9;
    });
  }

  function weldKey(p) { return Math.round(p[0] * 1e6) + ',' + Math.round(p[1] * 1e6) + ',' + Math.round(p[2] * 1e6); }

  function addWallQuad(out, interiorRef, P, Q, R, S) {
    pushTriangle(out, interiorRef, P, Q, R);
    pushTriangle(out, interiorRef, P, R, S);
  }

  // Extrudes one convex, planar polygon (a clipped module piece, already in
  // 3D, entirely on one face) into a closed box: top and bottom caps go
  // straight into `out`, but a wall is only ever a CANDIDATE at this point
  // -- pushed into `wallCandidates` and tallied in `wallEdgeCount` instead of
  // drawn immediately. `normals` is one outward direction per vertex -- the
  // same shared value for every vertex except a seam point, which carries a
  // blended normal instead (see blendedNormalAt), so two touching pieces
  // (same-face grid neighbors, or the two seam-clipped halves of one
  // straddling module) always compute IDENTICAL top and bottom points for
  // their shared wall.
  //
  // resolveWallCandidates (below), run once after every piece in the model
  // has been processed, keeps a wall only where wallEdgeCount says exactly
  // ONE piece claimed that edge (a genuine exterior boundary) and drops it
  // where two pieces claimed it: since both sides' top AND bottom edges
  // coincide there, the wall would be a zero-thickness internal partition
  // between two touching solids -- correct union boundary is to have NO
  // face there at all, just the two pieces' top caps (and separately their
  // bottom caps) meeting directly, which is exactly what dropping it leaves
  // behind. (An earlier version drew both copies and tried to cancel the
  // matching pair by opposite winding, but two independently-computed
  // per-box interior-reference points have no guarantee of landing on truly
  // opposite sides of a shared wall once a box has a mix of blended and
  // unblended vertex normals -- so the two copies could come out with the
  // SAME winding and simply fail to cancel, which is exactly the residual
  // non-manifold edges this replaced.)
  function buildExtrudedPrism(out, topPoly, normals, wallCandidates, wallEdgeCount) {
    const n = topPoly.length;
    const topOff = topPoly.map((p, i) => vAdd(p, vScale(normals[i], QR_EMBED_PROTRUDE_CM)));
    const bottomOff = topPoly.map((p, i) => vSub(p, vScale(normals[i], QR_EMBED_DEPTH_CM)));
    const interiorRef = vScale(topOff.concat(bottomOff).reduce((a, p) => vAdd(a, p), [0, 0, 0]), 1 / (2 * n));
    for (let i = 1; i < n - 1; i++) pushTriangle(out, interiorRef, topOff[0], topOff[i], topOff[i + 1]);
    for (let i = 1; i < n - 1; i++) pushTriangle(out, interiorRef, bottomOff[0], bottomOff[i], bottomOff[i + 1]);
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const edgeKey = [weldKey(topOff[i]), weldKey(topOff[j])].sort().join('|');
      wallEdgeCount.set(edgeKey, (wallEdgeCount.get(edgeKey) || 0) + 1);
      wallCandidates.push({ edgeKey, interiorRef, P: topOff[i], Q: topOff[j], R: bottomOff[j], S: bottomOff[i] });
    }
  }

  function resolveWallCandidates(out, wallCandidates, wallEdgeCount) {
    wallCandidates.forEach(({ edgeKey, interiorRef, P, Q, R, S }) => {
      if (wallEdgeCount.get(edgeKey) === 1) addWallQuad(out, interiorRef, P, Q, R, S);
    });
  }

  // moduleTargetCorners' 4 corners in (col,row) terms are
  // [bottom-left, bottom-right, top-right, top-left]; corner ci's DIAGONAL
  // grid neighbor (the one other module that touches ONLY at that single
  // point) is at MODULE_CORNER_DIAGONALS[ci], and its two EDGE neighbors
  // (which share a full edge -- and so a real wall -- through that same
  // point) are at (row+dr,col) and (row,col+dc).
  const MODULE_CORNER_DIAGONALS = [[-1, -1], [-1, 1], [1, 1], [1, -1]];
  const CHECKERBOARD_INSET_FRAC = 0.08; // fraction of the way from a touched corner toward this piece's own center

  function moduleHasFaceGeometry(qr, sizeCm, faceBoundaries, key, row, col, minAreaCm2) {
    if (row < 0 || row >= qr.size || col < 0 || col >= qr.size || !qr.modules.get(row, col)) return false;
    const clipped = clipPolygonConvex(moduleTargetCorners(qr, sizeCm, row, col), faceBoundaries[key]);
    return clipped.length >= 3 && Math.abs(polygonSignedArea(clipped.map((o) => o.pt))) >= minAreaCm2;
  }

  // Two modules whose squares touch ONLY diagonally (a checkerboard corner)
  // independently draw a box at that shared grid point -- and since both
  // sit on the same face with the same (unblended) normal, their two boxes'
  // corners land on EXACTLY the same 3D point, top and bottom alike. Unlike
  // a genuine shared edge (deliberately kept exact so resolveWallCandidates
  // can recognize and drop it), this coincidence isn't a duplicate of the
  // SAME wall -- it's two DIFFERENT walls (one from each module's own,
  // otherwise-unrelated box) that happen to share a vertical edge, which
  // still leaves that edge non-manifold (4 triangles meeting on it) no
  // matter how the walls are deduplicated. Nudging ONE corner of THIS
  // module's box slightly toward its own center -- only when the diagonal
  // neighbor is genuinely dark on this face AND neither of the two
  // in-between edge-neighbors also is (which would make this a real shared
  // edge instead) -- breaks the coincidence without touching any position
  // a real wall-match depends on.
  function applyCheckerboardInset(topPoly, qr, sizeCm, faceBoundaries, key, row, col, minAreaCm2) {
    const centroid = vScale(topPoly.reduce((a, p) => vAdd(a, p), [0, 0, 0]), 1 / topPoly.length);
    return topPoly.map((p, ci) => {
      const [dr, dc] = MODULE_CORNER_DIAGONALS[ci];
      const sharesRealEdge = moduleHasFaceGeometry(qr, sizeCm, faceBoundaries, key, row + dr, col, minAreaCm2)
        || moduleHasFaceGeometry(qr, sizeCm, faceBoundaries, key, row, col + dc, minAreaCm2);
      if (sharesRealEdge) return p; // a real edge-neighbor shares this corner -- keep it exact
      if (!moduleHasFaceGeometry(qr, sizeCm, faceBoundaries, key, row + dr, col + dc, minAreaCm2)) return p;
      return vAdd(p, vScale(vSub(centroid, p), CHECKERBOARD_INSET_FRAC));
    });
  }

  // One small extruded box per dark QR module (or per face-clipped piece of
  // one, for a module straddling a seam): top flush with (a hair outside)
  // the true outer surface, bottom QR_EMBED_DEPTH_CM further in along the
  // face's own outward normal. Reuses the exact same inverse-projection
  // math as the live texture bake and the flat print pieces, so the
  // embossed pattern matches what's shown/printed elsewhere pixel-for-pixel.
  // Touching boxes are fused via resolveWallCandidates below, so contiguous
  // dark blocks (finder patterns, timing patterns) come out as one
  // continuous solid rather than a pile of separately-walled boxes.
  function buildQrEmbedTriangles(s) {
    const v = computeVertices(s);
    const fc = faceCornerSets(v);
    const qr = ensureQR(s.qrText);
    const sizeCm = qrSizeCm(s);
    if (sizeCm <= 0 || qr.error) return [];
    const camHeight = revealCameraHeight(s);
    const interiorRef = [0, s.H * 0.3, 0];
    const faceKeys = s.facesMode === 4 ? ['front', 'back', 'left', 'right'] : ['front', 'back'];

    const faceNormals = {}, faceBoundaries = {};
    faceKeys.forEach((key) => {
      faceNormals[key] = computeFaceOutwardNormal(fc[key], interiorRef);
      faceBoundaries[key] = faceBoundaryUW(fc[key], camHeight);
    });

    const minAreaCm2 = 1e-6; // drops pure numerical slivers at clip boundaries; real geometry is filtered by pushTriangle's own degeneracy check
    const out = [];
    const wallCandidates = [];
    const wallEdgeCount = new Map();
    for (let row = 0; row < qr.size; row++) {
      for (let col = 0; col < qr.size; col++) {
        if (!qr.modules.get(row, col)) continue;
        const moduleQuad = moduleTargetCorners(qr, sizeCm, row, col);
        faceKeys.forEach((key) => {
          const clipped = clipPolygonConvex(moduleQuad, faceBoundaries[key]);
          if (clipped.length < 3 || Math.abs(polygonSignedArea(clipped.map((o) => o.pt))) < minAreaCm2) return;
          const topPoly = [], vertexNormals = [];
          for (const { pt: [u, w], boundary } of clipped) {
            const r = invertFaceProjection(fc[key], u, w, camHeight);
            if (!r) return;
            const ss = Math.min(1, Math.max(0, r.s)), tt = Math.min(1, Math.max(0, r.t));
            const c = fc[key];
            topPoly.push(bilinear(c.p00, c.p10, c.p11, c.p01, ss, tt, [0, 0, 0]));
            vertexNormals.push(boundary ? blendedNormalAt(faceKeys, faceBoundaries, faceNormals, key, u, w) : faceNormals[key]);
          }
          const insetPoly = clipped.length === 4 && clipped.every((o) => !o.boundary)
            ? applyCheckerboardInset(topPoly, qr, sizeCm, faceBoundaries, key, row, col, minAreaCm2)
            : topPoly;
          buildExtrudedPrism(out, insetPoly, vertexNormals, wallCandidates, wallEdgeCount);
        });
      }
    }
    resolveWallCandidates(out, wallCandidates, wallEdgeCount);
    return out;
  }

  // `tris` (a flat [x,y,z, x,y,z, ...] list, 3 floats per vertex, 3
  // vertices per triangle, no sharing) is the easy shape to build triangles
  // into, but leaving it unwelded means every face seam and every touching
  // pair of module-box vertices is duplicated -- geometrically coincident,
  // but topologically disconnected, which is again something a slicer's
  // manifold check can flag. Snapping each vertex to a coordinate key at
  // well below printing precision merges those duplicates into one real,
  // shared-index vertex, so adjoining faces/boxes actually share edges.
  function weldToIndexedMesh(tris, precision) {
    const scale = Math.pow(10, precision == null ? 6 : precision);
    const keyToIndex = new Map();
    const vertices = [];
    const nVerts = tris.length / 3;
    const remap = new Int32Array(nVerts);
    for (let i = 0; i < nVerts; i++) {
      const x = tris[i * 3], y = tris[i * 3 + 1], z = tris[i * 3 + 2];
      const key = Math.round(x * scale) + '_' + Math.round(y * scale) + '_' + Math.round(z * scale);
      let idx = keyToIndex.get(key);
      if (idx === undefined) {
        idx = vertices.length;
        vertices.push([x, y, z]);
        keyToIndex.set(key, idx);
      }
      remap[i] = idx;
    }
    const indices = [];
    for (let i = 0; i < nVerts; i += 3) {
      const a = remap[i], b = remap[i + 1], c = remap[i + 2];
      if (a === b || b === c || a === c) continue; // welding can turn a sliver into a degenerate triangle -- drop it same as pushTriangle does
      indices.push([a, b, c]);
    }
    return { vertices, indices };
  }

  function trianglesXmlObject(objId, name, pindex, mesh) {
    let vertsXml = '';
    mesh.vertices.forEach(([x, y, z]) => {
      vertsXml += `<vertex x="${(x * MM_PER_CM).toFixed(4)}" y="${(y * MM_PER_CM).toFixed(4)}" z="${(z * MM_PER_CM).toFixed(4)}"/>`;
    });
    let trisXml = '';
    mesh.indices.forEach(([a, b, c]) => {
      trisXml += `<triangle v1="${a}" v2="${b}" v3="${c}"/>`;
    });
    return `<object id="${objId}" type="model" name="${escapeXml(name)}" pid="1" pindex="${pindex}"><mesh><vertices>${vertsXml}</vertices><triangles>${trisXml}</triangles></mesh></object>`;
  }

  function build3mfModelXml(prismTris, qrTris) {
    const prismObj = trianglesXmlObject(10, 'Prism body', 0, weldToIndexedMesh(prismTris));
    const hasQr = qrTris.length > 0;
    const qrObj = hasQr ? trianglesXmlObject(20, 'QR embed body', 1, weldToIndexedMesh(qrTris)) : '';
    const components = hasQr
      ? '<component objectid="10"/><component objectid="20"/>'
      : '<component objectid="10"/>';
    return `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">
  <resources>
    <m:colorgroup id="1">
      <m:color color="${rgbToHex(BG_RGB)}"/>
      <m:color color="${rgbToHex(DARK_RGB)}"/>
    </m:colorgroup>
    ${prismObj}
    ${qrObj}
    <object id="1" type="model" name="${escapeXml('QR Hip Roof ' + paramCaptionShort(state))}">
      <components>${components}</components>
    </object>
  </resources>
  <build>
    <item objectid="1"/>
  </build>
</model>`;
  }

  // ---------- minimal ZIP (store, no compression) for the .3mf container ----------

  const CRC_TABLE = (() => {
    const table = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
      table[n] = c >>> 0;
    }
    return table;
  })();

  function crc32(bytes) {
    let crc = 0xFFFFFFFF;
    for (let i = 0; i < bytes.length; i++) crc = CRC_TABLE[(crc ^ bytes[i]) & 0xFF] ^ (crc >>> 8);
    return (crc ^ 0xFFFFFFFF) >>> 0;
  }

  function u16(n) { return [n & 0xFF, (n >>> 8) & 0xFF]; }
  function u32(n) { return [n & 0xFF, (n >>> 8) & 0xFF, (n >>> 16) & 0xFF, (n >>> 24) & 0xFF]; }

  function buildZip(files) {
    const encoder = new TextEncoder();
    const localParts = [];
    const centralParts = [];
    let offset = 0;
    files.forEach(({ name, data }) => {
      const nameBytes = encoder.encode(name);
      const crc = crc32(data);
      const size = data.length;
      const localHeader = new Uint8Array([
        0x50, 0x4b, 0x03, 0x04,
        ...u16(20), ...u16(0), ...u16(0), ...u16(0), ...u16(0x21),
        ...u32(crc), ...u32(size), ...u32(size),
        ...u16(nameBytes.length), ...u16(0),
      ]);
      localParts.push(localHeader, nameBytes, data);
      const centralHeader = new Uint8Array([
        0x50, 0x4b, 0x01, 0x02,
        ...u16(20), ...u16(20), ...u16(0), ...u16(0), ...u16(0), ...u16(0x21),
        ...u32(crc), ...u32(size), ...u32(size),
        ...u16(nameBytes.length), ...u16(0), ...u16(0), ...u16(0), ...u16(0), ...u32(0),
        ...u32(offset),
      ]);
      centralParts.push(centralHeader, nameBytes);
      offset += localHeader.length + nameBytes.length + size;
    });
    const centralStart = offset;
    const centralSize = centralParts.reduce((sum, p) => sum + p.length, 0);
    const eocd = new Uint8Array([
      0x50, 0x4b, 0x05, 0x06,
      ...u16(0), ...u16(0), ...u16(files.length), ...u16(files.length),
      ...u32(centralSize), ...u32(centralStart), ...u16(0),
    ]);
    return new Blob([...localParts, ...centralParts, eocd]);
  }

  function download3mf() {
    const sizeCm = qrSizeCm(state);
    const qr = ensureQR(state.qrText);
    if (sizeCm <= 0 || qr.error) return;
    const prismTris = buildPrismSolidTriangles(state);
    const qrTris = buildQrEmbedTriangles(state);
    const modelXml = build3mfModelXml(prismTris, qrTris);
    const contentTypesXml = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>`;
    const relsXml = `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>`;
    const encoder = new TextEncoder();
    const blob = buildZip([
      { name: '[Content_Types].xml', data: encoder.encode(contentTypesXml) },
      { name: '_rels/.rels', data: encoder.encode(relsXml) },
      { name: '3D/3dmodel.model', data: encoder.encode(modelXml) },
    ]);
    downloadBlob(`qr-hip-roof_${paramSlug(state)}.3mf`, blob, 'model/3mf');
  }

  // ---------- UI wiring ----------

  const DIM_IDS = ['L', 'W', 'R', 'H', 'viewDist'];

  function syncSliderOutputs() {
    DIM_IDS.forEach((id) => {
      const v = state[id];
      document.getElementById(id).value = v;
      const numInput = document.getElementById(id + '-num');
      if (document.activeElement !== numInput) numInput.value = v;
    });
  }

  function enforceRidgeConstraint() {
    const rSlider = document.getElementById('R');
    const rNum = document.getElementById('R-num');
    rSlider.max = state.L;
    rNum.max = state.L;
    if (state.R > state.L) {
      state.R = state.L;
    }
  }

  function bindUI() {
    DIM_IDS.forEach((id) => {
      const slider = document.getElementById(id);
      const numInput = document.getElementById(id + '-num');
      const min = parseFloat(slider.min), max = parseFloat(slider.max);

      const apply = (rawValue) => {
        let v = parseFloat(rawValue);
        if (!isFinite(v)) return;
        v = Math.min(Math.max(v, min), id === 'R' ? state.L : max);
        state[id] = v;
        if (id === 'L' || id === 'R') enforceRidgeConstraint();
        syncSliderOutputs();
        scheduleRebuild();
      };

      slider.addEventListener('input', () => apply(slider.value));
      numInput.addEventListener('input', () => apply(numInput.value));
      numInput.addEventListener('blur', () => { numInput.value = state[id]; });
    });

    document.getElementById('qrText').addEventListener('input', (e) => {
      state.qrText = e.target.value;
      scheduleRebuild();
    });

    document.querySelectorAll('#facesToggle button').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#facesToggle button').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        state.facesMode = parseInt(btn.dataset.mode, 10);
        scheduleRebuild();
      });
    });

    document.querySelectorAll('#pageSizeToggle button').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#pageSizeToggle button').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        state.pageSize = btn.dataset.size;
        updatePageCountReadout();
      });
    });

    document.querySelectorAll('#formatToggle button').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#formatToggle button').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        state.format = btn.dataset.format;
        updateFormatUI();
      });
    });

    document.getElementById('downloadBtn').addEventListener('click', () => {
      if (state.format === 'PDF') downloadPdf();
      else if (state.format === 'SVG') downloadSvg();
      else if (state.format === 'PNG') downloadPng();
      else download3mf();
    });

    document.getElementById('topViewBtn').addEventListener('click', viewFromTop);

    enforceRidgeConstraint();
    syncSliderOutputs();
    updateFormatUI();
  }

  function updateFormatUI() {
    const isPdf = state.format === 'PDF';
    const isModel = state.format === '3MF';
    document.getElementById('pageSizeField').hidden = !isPdf;
    document.getElementById('pageCountReadout').hidden = !isPdf;
    document.getElementById('pdfHint').hidden = !isPdf;
    document.getElementById('rasterHint').hidden = isPdf || isModel;
    document.getElementById('modelHint').hidden = !isModel;
    document.getElementById('downloadBtn').textContent = isPdf
      ? 'Download print-ready PDF'
      : isModel ? 'Download 3D model (.3mf)' : `Download ${state.format}`;
  }

  // ---------- init ----------

  document.addEventListener('DOMContentLoaded', () => {
    bindUI();
    initScene();
    rebuildAll();
  });
})();
