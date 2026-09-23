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

  const state = {
    L: 25, W: 25, R: 15, H: 5,
    viewDist: 40, // scan/reveal distance above the ridge top -- see revealCameraHeight()
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
    };
  }

  // P(s,t) = (1-s)(1-t)p00 + s(1-t)p10 + s t p11 + (1-s) t p01
  function bilinear(p00, p10, p11, p01, s, t, out) {
    const w00 = (1 - s) * (1 - t), w10 = s * (1 - t), w11 = s * t, w01 = (1 - s) * t;
    out[0] = w00 * p00[0] + w10 * p10[0] + w11 * p11[0] + w01 * p01[0];
    out[1] = w00 * p00[1] + w10 * p10[1] + w11 * p11[1] + w01 * p01[1];
    out[2] = w00 * p00[2] + w10 * p10[2] + w11 * p11[2] + w01 * p01[2];
    return out;
  }

  function buildFaceGeometry(corners, n) {
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
        uvs[ui++] = s; uvs[ui++] = t;
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

    ['front', 'back', 'left', 'right'].forEach((name) => {
      const geo = new THREE.BufferGeometry();
      const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });
      const mesh = new THREE.Mesh(geo, mat);
      scene.add(mesh);
      meshes[name] = mesh;
    });

    // No ground plane / grid -- the prism sits in open space so orbiting
    // all the way to directly overhead (or any other angle) is never
    // visually or functionally blocked by a "floor".
    const perspCanvas = document.getElementById('perspCanvas');
    perspRenderer = new THREE.WebGLRenderer({ canvas: perspCanvas, antialias: true });
    perspRenderer.outputEncoding = THREE.sRGBEncoding;
    perspCamera = new THREE.PerspectiveCamera(DEFAULT_FOV, 1, 1, 5000);
    orbitControls = new THREE.OrbitControls(perspCamera, perspCanvas);
    orbitControls.enableDamping = true;
    orbitControls.dampingFactor = 0.08;
    orbitControls.minDistance = 10;
    orbitControls.maxDistance = 2000;
    orbitControls.minPolarAngle = 0;       // straight down from directly above...
    orbitControls.maxPolarAngle = Math.PI; // ...through straight up from directly below

    const topCanvas = document.getElementById('topCanvas');
    topRenderer = new THREE.WebGLRenderer({ canvas: topCanvas, antialias: true });
    topRenderer.outputEncoding = THREE.sRGBEncoding;
    // Also a real PerspectiveCamera: the "reveal" is now defined as the view
    // from a camera state.viewDist above the ridge top (see
    // revealCameraHeight), not an idealized orthographic one, so this pane
    // has to use the same finite-distance camera to match what was
    // actually baked into the textures.
    topCamera = new THREE.PerspectiveCamera(TOP_VIEW_FOV, 1, 1, 5000);

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
  // top" preset -- it moves the orbit camera to the reveal position but
  // leaves OrbitControls fully live, so the user can immediately keep
  // orbiting/zooming away from there.
  function viewFromTop() {
    orbitControls.target.set(0, 0, 0);
    positionRevealCamera(perspCamera, true);
    orbitControls.update();
  }

  function frameCameras() {
    const maxDim = Math.max(state.L, state.W, state.H * 2, 20);
    const dist = maxDim * 1.7;
    perspCamera.position.set(dist * 0.55, dist * 0.6, dist * 0.85);
    const focusY = state.H * 0.3;
    perspCamera.lookAt(0, focusY, 0);
    orbitControls.target.set(0, focusY, 0);
    orbitControls.update();

    positionRevealCamera(topCamera, true);
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
    }
    if (resizeRendererToDisplaySize(topRenderer)) {
      topCamera.aspect = topRenderer.domElement.clientWidth / topRenderer.domElement.clientHeight;
      topCamera.updateProjectionMatrix();
    }

    perspRenderer.render(scene, perspCamera);
    topRenderer.render(scene, topCamera);
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
        tex.needsUpdate = true;
        mesh.material.map = tex;
        mesh.material.color.set(0xffffff);
      } else {
        mesh.material.color.set(0xd8d8d2);
      }
      mesh.material.needsUpdate = true;
    });

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
      else downloadPng();
    });

    document.getElementById('topViewBtn').addEventListener('click', viewFromTop);

    enforceRidgeConstraint();
    syncSliderOutputs();
    updateFormatUI();
  }

  function updateFormatUI() {
    const isPdf = state.format === 'PDF';
    document.getElementById('pageSizeField').hidden = !isPdf;
    document.getElementById('pageCountReadout').hidden = !isPdf;
    document.getElementById('pdfHint').hidden = !isPdf;
    document.getElementById('rasterHint').hidden = isPdf;
    document.getElementById('downloadBtn').textContent =
      isPdf ? 'Download print-ready PDF' : `Download ${state.format}`;
  }

  // ---------- init ----------

  document.addEventListener('DOMContentLoaded', () => {
    bindUI();
    initScene();
    rebuildAll();
  });
})();
