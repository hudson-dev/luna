/**
 * Interactive Three.js parafoil flight viewer.
 * Expects window.FLIGHT_LOG (JSON object from flight_log.py).
 *
 * Scene frame: Three.js Y-up ENU — x=East, y=Up, z=North.
 * Body frame: FRD (x forward, y right, z down). Quaternions in the log are
 * body→NED (scalar-first); we convert to body→scene via NED→ENU.
 */
(function () {
  "use strict";

  const LOG = window.FLIGHT_LOG;
  if (!LOG) {
    document.body.innerHTML = "<p style='padding:2rem;font-family:sans-serif'>Missing FLIGHT_LOG.</p>";
    return;
  }

  const PHASE_COLORS = {
    HOMING: 0x3b82f6,
    LOITER: 0xf59e0b,
    EXTEND: 0xa855f7,
    APPROACH: 0x22c55e,
    FLARE: 0xef4444,
  };
  const CANOPY_COLOR = 0xe8641e;
  const CANOPY_UNDER = 0xc44e12;
  const PAYLOAD_COLOR = 0x37474f;
  const LINE_COLOR = 0x555555;
  const BRAKE_L = 0xd62728;
  const BRAKE_R = 0x2ca02c;

  const TE_DEFLECT_MAX = (40 * Math.PI) / 180;
  const HINGE_FRACTION = 0.30;
  const N_RIB = 13;
  const N_CHORD = 11;
  const NED_TO_SCENE = [
    [0, 1, 0],
    [0, 0, -1],
    [1, 0, 0],
  ];

  // ---------- math helpers ----------
  function quatNedToScene(qw, qx, qy, qz) {
    // Geometry is stored in local scene axes via bodyToScene(), so convert
    // the body→NED rotation into that same basis:
    //
    //   R_scene = T * R_ned_body * T^-1
    //
    // where T maps (N,E,D) to Three.js (E,Up,N).  Using only T*R would apply
    // T twice to the already-converted geometry and make the follow-camera's
    // local "back" vector track body vertical instead of body aft.
    const R = dcmFromQuat(qw, qx, qy, qz);
    const T = NED_TO_SCENE;
    const M = mat3Multiply(mat3Multiply(T, R), mat3Transpose(T));
    return mat3ToQuat(M);
  }

  function mat3Multiply(A, B) {
    return A.map((row) =>
      B[0].map((_, j) =>
        row.reduce((sum, value, k) => sum + value * B[k][j], 0)
      )
    );
  }

  function mat3Transpose(A) {
    return A[0].map((_, j) => A.map((row) => row[j]));
  }

  function dcmFromQuat(w, x, y, z) {
    return [
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ];
  }

  function mat3ToQuat(M) {
    // robust conversion, returns THREE.Quaternion-compatible {x,y,z,w}
    const m00 = M[0][0], m01 = M[0][1], m02 = M[0][2];
    const m10 = M[1][0], m11 = M[1][1], m12 = M[1][2];
    const m20 = M[2][0], m21 = M[2][1], m22 = M[2][2];
    const tr = m00 + m11 + m22;
    let w, x, y, z;
    if (tr > 0) {
      const s = 0.5 / Math.sqrt(tr + 1);
      w = 0.25 / s;
      x = (m21 - m12) * s;
      y = (m02 - m20) * s;
      z = (m10 - m01) * s;
    } else if (m00 > m11 && m00 > m22) {
      const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
      w = (m21 - m12) / s;
      x = 0.25 * s;
      y = (m01 + m10) / s;
      z = (m02 + m20) / s;
    } else if (m11 > m22) {
      const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
      w = (m02 - m20) / s;
      x = (m01 + m10) / s;
      y = 0.25 * s;
      z = (m12 + m21) / s;
    } else {
      const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
      w = (m10 - m01) / s;
      x = (m02 + m20) / s;
      y = (m12 + m21) / s;
      z = 0.25 * s;
    }
    return { x, y, z, w };
  }

  function bodyToScene(p) {
    // body FRD → scene ENU when attitude is identity (NED=body): (x,y,z)_frd
    // maps via NED→ENU as (y, -z, x)
    return new THREE.Vector3(p[1], -p[2], p[0]);
  }

  /** Flight-log ENU (E, N, U) → Three.js scene (x=E, y=Up, z=N). */
  function enuToScene(p) {
    return new THREE.Vector3(p[0], p[2], p[1]);
  }

  // ---------- canopy geometry (matches viz/rig_geometry.py) ----------
  function canopyGrid(veh, brakeL, brakeR) {
    const b = veh.span, c = veh.chord;
    const arc = veh.arc_ratio * b;
    const R = ((b / 2) ** 2 + arc ** 2) / (2 * Math.max(arc, 1e-6));
    const thM = Math.asin((b / 2) / R);
    const zC = veh.r_canopy[2];
    const thick = veh.thickness;

    const thetas = [];
    for (let i = 0; i < N_RIB; i++) thetas.push(-thM + (2 * thM * i) / (N_RIB - 1));
    const s = [];
    for (let j = 0; j < N_CHORD; j++) s.push(j / (N_CHORD - 1));
    const xChord = s.map((si) => c / 2 - si * c);
    const camber = s.map((si) => -0.08 * c * Math.sin(Math.PI * si));
    const hingeS = 1.0 - HINGE_FRACTION;
    const xHinge = c / 2 - hingeS * c;
    const camberHinge = -0.08 * c * Math.sin(Math.PI * hingeS);

    const upper = [];
    const lower = [];
    for (let i = 0; i < N_RIB; i++) {
      const th = thetas[i];
      const y = R * Math.sin(th);
      const zRib = zC + R * (1 - Math.cos(th));
      const yHat = y / (b / 2);
      const norm =
        brakeL * Math.max(0, -yHat) + brakeR * Math.max(0, yHat);
      const delta =
        TE_DEFLECT_MAX * norm * (0.25 + 0.75 * Math.abs(yHat));
      const cd = Math.cos(delta), sd = Math.sin(delta);
      // local thickness normal (approx upward in body = -z)
      const halfT = (thick * 0.5) * (0.35 + 0.65 * Math.sin(Math.PI * 0.5)); // mid-chord bulge later

      upper[i] = [];
      lower[i] = [];
      for (let j = 0; j < N_CHORD; j++) {
        let x = xChord[j], dz = camber[j];
        if (x < xHinge) {
          const dx = x - xHinge;
          const dzh = dz - camberHinge;
          x = xHinge + dx * cd + dzh * sd;
          dz = camberHinge - dx * sd + dzh * cd;
        }
        // cell thickness profile: blunt LE, taper TE
        const tProf = Math.sin(Math.PI * Math.min(s[j], 0.92)) * 0.92;
        const ht = (thick * 0.5) * tProf;
        // inflate along local "up" (-z body) with slight camber offset
        upper[i][j] = [x, y, zRib + dz - ht];
        lower[i][j] = [x, y, zRib + dz + ht * 0.55];
      }
    }
    return { upper, lower };
  }

  function gridToGeometry(upper, lower) {
    const positions = [];
    const colors = [];
    const indices = [];
    const uvs = [];

    function cellColor(i, j, under) {
      // alternate spanwise cells; darken underside
      const stripe = (Math.floor(i) % 2 === 0) ? 1.0 : 0.82;
      const base = under ? [0.72, 0.28, 0.08] : [0.91, 0.39, 0.12];
      // slight LE brightening
      const le = 1.0 + 0.08 * (1 - j / (N_CHORD - 1));
      return [base[0] * stripe * le, base[1] * stripe * le, base[2] * stripe * le];
    }

    function pushV(p, u, v, i, j, under) {
      positions.push(p[1], -p[2], p[0]); // body FRD → local scene axes
      uvs.push(u, v);
      const c = cellColor(i, j, under);
      colors.push(c[0], c[1], c[2]);
    }

    // upper surface
    for (let i = 0; i < N_RIB; i++) {
      for (let j = 0; j < N_CHORD; j++) {
        pushV(upper[i][j], j / (N_CHORD - 1), i / (N_RIB - 1), i, j, false);
      }
    }
    for (let i = 0; i < N_RIB - 1; i++) {
      for (let j = 0; j < N_CHORD - 1; j++) {
        const a = i * N_CHORD + j;
        const b = a + N_CHORD;
        const c = a + 1;
        const d = b + 1;
        indices.push(a, b, c, c, b, d);
      }
    }

    // lower surface (reversed winding)
    const baseL = N_RIB * N_CHORD;
    for (let i = 0; i < N_RIB; i++) {
      for (let j = 0; j < N_CHORD; j++) {
        pushV(lower[i][j], j / (N_CHORD - 1), i / (N_RIB - 1), i, j, true);
      }
    }
    for (let i = 0; i < N_RIB - 1; i++) {
      for (let j = 0; j < N_CHORD - 1; j++) {
        const a = baseL + i * N_CHORD + j;
        const b = a + N_CHORD;
        const c = a + 1;
        const d = b + 1;
        indices.push(a, c, b, c, d, b);
      }
    }

    // cell walls (ribs) between upper/lower
    for (let i = 0; i < N_RIB; i++) {
      for (let j = 0; j < N_CHORD - 1; j++) {
        const u0 = i * N_CHORD + j;
        const u1 = u0 + 1;
        const l0 = baseL + u0;
        const l1 = baseL + u1;
        indices.push(u0, l0, u1, u1, l0, l1);
      }
    }

    // LE / TE closing
    for (let i = 0; i < N_RIB - 1; i++) {
      let u0 = i * N_CHORD, u1 = (i + 1) * N_CHORD;
      let l0 = baseL + u0, l1 = baseL + u1;
      indices.push(u0, u1, l0, l0, u1, l1);
      u0 = i * N_CHORD + (N_CHORD - 1);
      u1 = (i + 1) * N_CHORD + (N_CHORD - 1);
      l0 = baseL + u0;
      l1 = baseL + u1;
      indices.push(u0, l0, u1, u1, l0, l1);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    geo.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    return geo;
  }

  function makeCellLines(upper) {
    const pts = [];
    // chordwise ribs
    for (let i = 0; i < N_RIB; i++) {
      for (let j = 0; j < N_CHORD; j++) {
        const p = upper[i][j];
        pts.push(new THREE.Vector3(p[1], -p[2], p[0]));
      }
      pts.push(null);
    }
    // LE + TE
    for (const j of [0, N_CHORD - 1]) {
      for (let i = 0; i < N_RIB; i++) {
        const p = upper[i][j];
        pts.push(new THREE.Vector3(p[1], -p[2], p[0]));
      }
      pts.push(null);
    }
    // spanwise seams at a few chord stations
    for (const jf of [0.25, 0.5, 0.75]) {
      const j = Math.round(jf * (N_CHORD - 1));
      for (let i = 0; i < N_RIB; i++) {
        const p = upper[i][j];
        pts.push(new THREE.Vector3(p[1], -p[2], p[0]));
      }
      pts.push(null);
    }
    return pts;
  }

  function makeSuspension(veh, upper) {
    const p0 = veh.r_payload;
    const riserTop = {
      L: [0.02, -0.10, p0[2] - 0.10],
      R: [0.02, +0.10, p0[2] - 0.10],
    };
    const conf = {
      L: [
        riserTop.L[0] + 0.62 * (veh.r_canopy[0] - riserTop.L[0]),
        riserTop.L[1] + 0.62 * (veh.r_canopy[1] - riserTop.L[1]) - 0.12,
        riserTop.L[2] + 0.62 * (veh.r_canopy[2] - riserTop.L[2]),
      ],
      R: [
        riserTop.R[0] + 0.62 * (veh.r_canopy[0] - riserTop.R[0]),
        riserTop.R[1] + 0.62 * (veh.r_canopy[1] - riserTop.R[1]) + 0.12,
        riserTop.R[2] + 0.62 * (veh.r_canopy[2] - riserTop.R[2]),
      ],
    };
    const segs = [];
    // A / B / C cascade rows
    for (const jFrac of [0.10, 0.35, 0.58]) {
      const j = Math.round(jFrac * (N_CHORD - 1));
      for (let i = 0; i < N_RIB; i += 2) {
        const side = upper[i][j][1] < 0 ? "L" : "R";
        segs.push([upper[i][j], conf[side]]);
      }
    }
    for (const side of ["L", "R"]) {
      segs.push([conf[side], riserTop[side]]);
    }
    return { segs, conf, riserTop, servoC: {
      L: [-0.08, -0.085, p0[2] - 0.02],
      R: [-0.08, +0.085, p0[2] - 0.02],
    }};
  }

  function makeBrakeLines(upper, conf, servoC, spoolL, spoolR) {
    function side(which, spoolAngle) {
      const sgn = which === "L" ? -1 : 1;
      const iAtt = which === "L" ? 1 : N_RIB - 2;
      const te = upper[iAtt][N_CHORD - 1];
      const guide = [
        conf[which][0] - 0.05,
        conf[which][1] + sgn * 0.03,
        conf[which][2] + 0.05,
      ];
      const spool = servoC[which];
      const arm = [
        spool[0] + 0.055 * Math.cos(spoolAngle),
        spool[1],
        spool[2] - 0.055 * Math.sin(spoolAngle),
      ];
      return { line: [te, guide, spool], arm: [spool, arm], spool, armTip: arm, angle: spoolAngle };
    }
    return { L: side("L", spoolL), R: side("R", spoolR) };
  }

  function linesToPositions(segs) {
    // segs: array of polylines (arrays of body FRD points)
    const arr = [];
    for (const seg of segs) {
      for (let i = 0; i < seg.length; i++) {
        const p = seg[i];
        arr.push(p[1], -p[2], p[0]);
        if (i > 0 && i < seg.length - 1) {
          // duplicate interior for THREE.LineSegments pairs? Use Line with gaps via NaN
        }
      }
    }
    return arr;
  }

  function polylinePositions(polylines) {
    // Return Float32 for Line with breaks via restart (push same point pairs for LineSegments)
    const pos = [];
    for (const pl of polylines) {
      for (let i = 0; i < pl.length - 1; i++) {
        const a = pl[i], b = pl[i + 1];
        pos.push(a[1], -a[2], a[0], b[1], -b[2], b[0]);
      }
    }
    return pos;
  }

  // ---------- scene setup ----------
  const canvas = document.getElementById("c");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xb8c9d9);
  scene.fog = new THREE.Fog(0xb8c9d9, 800, 4200);

  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.05, 8000);
  camera.position.set(180, 140, -220);

  const controls = new (window.OrbitControls || THREE.OrbitControls)(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.maxPolarAngle = Math.PI * 0.49;
  controls.minDistance = 0.4;
  controls.maxDistance = 3500;

  // lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const sun = new THREE.DirectionalLight(0xfff2dd, 1.15);
  sun.position.set(200, 400, 120);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.near = 10;
  sun.shadow.camera.far = 1200;
  sun.shadow.camera.left = -200;
  sun.shadow.camera.right = 200;
  sun.shadow.camera.top = 200;
  sun.shadow.camera.bottom = -200;
  scene.add(sun);
  scene.add(new THREE.HemisphereLight(0x9ec9ff, 0x8a7a5c, 0.35));

  // ground
  const groundSize = 3000;
  const groundGeo = new THREE.PlaneGeometry(groundSize, groundSize);
  const groundMat = new THREE.MeshStandardMaterial({
    color: 0x6b8f5e,
    roughness: 0.95,
    metalness: 0.0,
  });
  const ground = new THREE.Mesh(groundGeo, groundMat);
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);

  // subtle grid
  const grid = new THREE.GridHelper(2000, 40, 0x4a5d3a, 0x5a7050);
  grid.position.y = 0.02;
  scene.add(grid);

  // pad / target
  const tgt = LOG.scenario.target_enu;
  const padGroup = new THREE.Group();
  padGroup.position.set(tgt[0], 0.05, tgt[1]);
  const padMat = new THREE.MeshStandardMaterial({ color: 0xc62828, roughness: 0.25 });
  const padRing = new THREE.Mesh(new THREE.RingGeometry(4, 8, 48), padMat);
  padRing.rotation.x = -Math.PI / 2;
  padGroup.add(padRing);
  const padInner = new THREE.Mesh(new THREE.CircleGeometry(3.5, 32), new THREE.MeshStandardMaterial({ color: 0xffffff }));
  padInner.rotation.x = -Math.PI / 2;
  padInner.position.y = 0.01;
  padGroup.add(padInner);
  const padX = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.08, 7), new THREE.MeshStandardMaterial({ color: 0xc62828 }));
  padGroup.add(padX);
  const padX2 = padX.clone();
  padX2.rotation.y = Math.PI / 2;
  padGroup.add(padX2);
  scene.add(padGroup);

  // trajectory ribbon
  const frames = LOG.frames;
  const n = frames.t.length;
  const trajPts = [];
  const trajColors = [];
  for (let i = 0; i < n; i++) {
    const p = frames.pos_enu[i];
    trajPts.push(p[0], p[2], p[1]); // E, Up, N
    const col = new THREE.Color(PHASE_COLORS[frames.phase[i]] || 0x888888);
    trajColors.push(col.r, col.g, col.b);
  }
  const trajGeo = new THREE.BufferGeometry();
  trajGeo.setAttribute("position", new THREE.Float32BufferAttribute(trajPts, 3));
  trajGeo.setAttribute("color", new THREE.Float32BufferAttribute(trajColors, 3));
  const trajLine = new THREE.Line(
    trajGeo,
    new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 })
  );
  scene.add(trajLine);

  // ground track
  const gtPts = [];
  for (let i = 0; i < n; i++) {
    const p = frames.pos_enu[i];
    gtPts.push(p[0], 0.08, p[1]); // E, ground, N
  }
  const gtGeo = new THREE.BufferGeometry();
  gtGeo.setAttribute("position", new THREE.Float32BufferAttribute(gtPts, 3));
  const groundTrack = new THREE.Line(gtGeo, new THREE.LineBasicMaterial({ color: 0x333333, transparent: true, opacity: 0.45 }));
  scene.add(groundTrack);

  // wind vectors (sparse)
  const windGroup = new THREE.Group();
  scene.add(windGroup);
  const windArrowGeo = new THREE.ConeGeometry(0.35, 1.2, 8);
  const windShaftGeo = new THREE.CylinderGeometry(0.08, 0.08, 1, 6);
  const windMat = new THREE.MeshStandardMaterial({ color: 0x1565c0, roughness: 0.4 });

  function rebuildWindArrows(stride) {
    while (windGroup.children.length) {
      const c = windGroup.children.pop();
      c.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
      });
    }
    for (let i = 0; i < n; i += stride) {
      const w = frames.wind_enu[i];
      const spd = Math.hypot(w[0], w[1], w[2]);
      if (spd < 0.4) continue;
      const p = frames.pos_enu[i];
      const g = new THREE.Group();
      const len = Math.min(spd * 3.5, 28);
      const shaft = new THREE.Mesh(windShaftGeo, windMat);
      shaft.scale.set(1, len, 1);
      shaft.position.y = len / 2;
      const tip = new THREE.Mesh(windArrowGeo, windMat);
      tip.position.y = len + 0.4;
      g.add(shaft, tip);
      // default cone points +Y; aim toward wind direction in scene (E, Up, N)
      const dir = new THREE.Vector3(w[0], w[2], w[1]).normalize();
      g.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
      g.position.copy(enuToScene(p));
      g.userData.frame = i;
      windGroup.add(g);
    }
  }
  rebuildWindArrows(Math.max(1, Math.floor(n / 40)));

  // ---------- vehicle rig ----------
  const veh = LOG.vehicle;
  const vehicle = new THREE.Group();
  scene.add(vehicle);

  const canopyMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    vertexColors: true,
    roughness: 0.78,
    metalness: 0.02,
    side: THREE.DoubleSide,
  });
  const canopyMesh = new THREE.Mesh(new THREE.BufferGeometry(), canopyMat);
  canopyMesh.castShadow = true;
  canopyMesh.receiveShadow = true;
  vehicle.add(canopyMesh);

  const ribMat = new THREE.LineBasicMaterial({ color: 0x5a2a0a, transparent: true, opacity: 0.55 });
  const ribLines = new THREE.LineSegments(new THREE.BufferGeometry(), ribMat);
  vehicle.add(ribLines);

  const suspMat = new THREE.LineBasicMaterial({ color: LINE_COLOR });
  const suspLines = new THREE.LineSegments(new THREE.BufferGeometry(), suspMat);
  vehicle.add(suspLines);

  const brakeLMat = new THREE.LineBasicMaterial({ color: BRAKE_L });
  const brakeRMat = new THREE.LineBasicMaterial({ color: BRAKE_R });
  const brakeLLines = new THREE.LineSegments(new THREE.BufferGeometry(), brakeLMat);
  const brakeRLines = new THREE.LineSegments(new THREE.BufferGeometry(), brakeRMat);
  vehicle.add(brakeLLines, brakeRLines);

  // payload body
  const p0 = veh.r_payload;
  const payMesh = new THREE.Mesh(
    new THREE.BoxGeometry(0.50, 0.14, 0.12),
    new THREE.MeshStandardMaterial({ color: PAYLOAD_COLOR, roughness: 0.55, metalness: 0.2 })
  );
  // box in local scene axes: size along (body x→z scene, body z→-y, body y→x)
  // Place using bodyToScene of payload center
  const payC = bodyToScene([0.05, 0.0, p0[2] + 0.05]);
  payMesh.position.copy(payC);
  // orient box: body axes mapped — rotate so box X along body X (scene Z), etc.
  // Easier: build payload in body then convert verts — use Euler for FRD→scene identity
  payMesh.rotation.set(0, 0, 0);
  // Remap geometry: Three box is X,Y,Z. We want extents in body (0.50, 0.12, 0.14) mapped to scene.
  payMesh.geometry.dispose();
  payMesh.geometry = new THREE.BoxGeometry(0.12, 0.14, 0.50); // (body y, -body z dim, body x)
  payMesh.castShadow = true;
  vehicle.add(payMesh);

  // servo housings + spool drums
  const servoMeshes = { L: null, R: null };
  const spoolMeshes = { L: null, R: null };
  const armMeshes = { L: null, R: null };
  for (const side of ["L", "R"]) {
    const sgn = side === "L" ? -1 : 1;
    const sc = [-0.08, sgn * 0.085, p0[2] - 0.02];
    const house = new THREE.Mesh(
      new THREE.BoxGeometry(0.05, 0.07, 0.07),
      new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.4, metalness: 0.3 })
    );
    house.position.copy(bodyToScene(sc));
    vehicle.add(house);
    servoMeshes[side] = house;

    const spool = new THREE.Mesh(
      new THREE.CylinderGeometry(veh.spool_radius * 1.8, veh.spool_radius * 1.8, 0.035, 24),
      new THREE.MeshStandardMaterial({ color: side === "L" ? BRAKE_L : BRAKE_R, metalness: 0.5, roughness: 0.35 })
    );
    // cylinder default Y-axis; align with body Y (scene X)
    spool.rotation.z = Math.PI / 2;
    spool.position.copy(bodyToScene(sc));
    vehicle.add(spool);
    spoolMeshes[side] = spool;

    const arm = new THREE.Mesh(
      new THREE.BoxGeometry(0.01, 0.01, 0.055),
      new THREE.MeshStandardMaterial({ color: 0xeeeeee, metalness: 0.6, roughness: 0.3 })
    );
    arm.geometry.translate(0, 0, 0.0275);
    arm.position.copy(bodyToScene(sc));
    vehicle.add(arm);
    armMeshes[side] = arm;
  }

  let lastSusp = null;

  function updateRig(brakeL, brakeR, spoolL, spoolR) {
    const { upper, lower } = canopyGrid(veh, brakeL, brakeR);
    canopyMesh.geometry.dispose();
    canopyMesh.geometry = gridToGeometry(upper, lower);

    // ribs
    const ribPts = makeCellLines(upper);
    const ribPos = [];
    let chain = [];
    function flush() {
      for (let i = 0; i < chain.length - 1; i++) {
        ribPos.push(chain[i].x, chain[i].y, chain[i].z, chain[i + 1].x, chain[i + 1].y, chain[i + 1].z);
      }
      chain = [];
    }
    for (const p of ribPts) {
      if (p === null) flush();
      else chain.push(p);
    }
    flush();
    ribLines.geometry.dispose();
    ribLines.geometry = new THREE.BufferGeometry();
    ribLines.geometry.setAttribute("position", new THREE.Float32BufferAttribute(ribPos, 3));

    const { segs, conf, servoC } = makeSuspension(veh, upper);
    lastSusp = { conf, servoC };
    const susPos = polylinePositions(segs);
    suspLines.geometry.dispose();
    suspLines.geometry = new THREE.BufferGeometry();
    suspLines.geometry.setAttribute("position", new THREE.Float32BufferAttribute(susPos, 3));

    const brakes = makeBrakeLines(upper, conf, servoC, spoolL, spoolR);
    for (const [side, lines, mesh] of [
      ["L", brakes.L, brakeLLines],
      ["R", brakes.R, brakeRLines],
    ]) {
      const pos = polylinePositions([lines.line, lines.arm]);
      mesh.geometry.dispose();
      mesh.geometry = new THREE.BufferGeometry();
      mesh.geometry.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    }

    // spool spin: arm rotates in body x-z plane → scene z-y
    for (const side of ["L", "R"]) {
      const ang = side === "L" ? spoolL : spoolR;
      const sc = servoC[side];
      armMeshes[side].position.copy(bodyToScene(sc));
      // arm along body +x at ang=0; rotation about body +y (scene +x)
      armMeshes[side].rotation.set(0, 0, 0);
      armMeshes[side].quaternion.setFromAxisAngle(new THREE.Vector3(1, 0, 0), -ang);
      spoolMeshes[side].rotation.x = -ang; // visual spin on cylinder after z-rot
      spoolMeshes[side].rotation.z = Math.PI / 2;
    }
  }

  // ---------- playback state ----------
  let idx = 0;
  let playing = true;
  let speed = 1.0;
  let camMode = "follow"; // free | follow | rig
  let showWind = true;
  let showTraj = true;
  let manualRendering = false;
  let accum = 0;
  const worldUp = new THREE.Vector3(0, 1, 0);
  const followHeading = new THREE.Vector3(0, 0, 1);

  const elPlay = document.getElementById("btn-play");
  const elScrub = document.getElementById("scrub");
  const elTime = document.getElementById("time-label");
  const elSpeed = document.getElementById("speed");
  const elPhase = document.getElementById("phase-label");
  const elBrake = document.getElementById("brake-label");
  const elMiss = document.getElementById("miss-label");

  elScrub.max = String(n - 1);
  elMiss.textContent =
    `miss ${LOG.summary.miss_distance_m.toFixed(1)} m · ` +
    `${LOG.summary.td_ground_speed_mps.toFixed(1)} m/s gs · ` +
    `flare ${LOG.summary.flare_triggered ? "yes" : "no"}`;

  document.getElementById("title").textContent =
    `Parafoil 3D — ${LOG.scenario.name}`;
  document.getElementById("subtitle").textContent = LOG.scenario.description;

  function setFrame(i) {
    idx = Math.max(0, Math.min(n - 1, i | 0));
    elScrub.value = String(idx);
    const t = frames.t[idx];
    const bl = frames.line_l[idx] / veh.line_travel_max;
    const br = frames.line_r[idx] / veh.line_travel_max;
    const sl = frames.line_l[idx] / veh.spool_radius;
    const sr = frames.line_r[idx] / veh.spool_radius;
    updateRig(bl, br, sl, sr);

    const p = frames.pos_enu[idx];
    const q = frames.quat_wxyz[idx];
    const qs = quatNedToScene(q[0], q[1], q[2], q[3]);
    vehicle.position.copy(enuToScene(p));
    vehicle.quaternion.set(qs.x, qs.y, qs.z, qs.w);

    elTime.textContent = `t = ${t.toFixed(1)} s`;
    elPhase.textContent = frames.phase[idx];
    elPhase.style.color = "#" + (PHASE_COLORS[frames.phase[idx]] || 0x888888).toString(16).padStart(6, "0");
    elBrake.textContent =
      `brake L ${(bl * 100).toFixed(0)}% / R ${(br * 100).toFixed(0)}% · ` +
      `line ${(frames.line_l[idx] * 100).toFixed(1)} / ${(frames.line_r[idx] * 100).toFixed(1)} cm`;

    // highlight nearest wind arrows
    windGroup.visible = showWind;
    trajLine.visible = showTraj;
    groundTrack.visible = showTraj;
  }

  function updateCamera(dt) {
    const p = vehicle.position;
    if (camMode === "follow") {
      // Local +Z is body-forward after bodyToScene(). Project it onto the
      // horizontal plane so roll/pitch cannot orbit the chase camera.
      followHeading.set(0, 0, 1).applyQuaternion(vehicle.quaternion);
      followHeading.y = 0;
      if (followHeading.lengthSq() < 1e-8) followHeading.set(0, 0, 1);
      followHeading.normalize();
      const desired = p.clone()
        .addScaledVector(followHeading, -28)
        .addScaledVector(worldUp, 12);
      camera.position.lerp(desired, 1 - Math.exp(-3 * dt));
      controls.target.lerp(p.clone().add(new THREE.Vector3(0, 1.2, 0)), 1 - Math.exp(-4 * dt));
    } else if (camMode === "rig") {
      // frame full canopy + payload + spools (body FRD → scene, then vehicle pose)
      const eyeBody = bodyToScene([2.4, 1.6, -0.35]);
      const lookBody = bodyToScene([0.0, 0.0, veh.r_canopy[2] * 0.45]);
      const eye = eyeBody.applyQuaternion(vehicle.quaternion).add(p);
      const look = lookBody.applyQuaternion(vehicle.quaternion).add(p);
      camera.position.lerp(eye, 1 - Math.exp(-4 * dt));
      controls.target.lerp(look, 1 - Math.exp(-5 * dt));
    }
    // free: OrbitControls only
  }

  elPlay.addEventListener("click", () => {
    playing = !playing;
    elPlay.textContent = playing ? "Pause" : "Play";
  });
  elScrub.addEventListener("input", () => {
    playing = false;
    elPlay.textContent = "Play";
    setFrame(+elScrub.value);
  });
  elSpeed.addEventListener("change", () => {
    speed = parseFloat(elSpeed.value);
  });
  document.querySelectorAll("[data-cam]").forEach((btn) => {
    btn.addEventListener("click", () => {
      camMode = btn.getAttribute("data-cam");
      document.querySelectorAll("[data-cam]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      if (camMode === "free") {
        // leave camera where it is
      }
    });
  });
  document.getElementById("tog-wind").addEventListener("change", (e) => {
    showWind = e.target.checked;
    windGroup.visible = showWind;
  });
  document.getElementById("tog-traj").addEventListener("change", (e) => {
    showTraj = e.target.checked;
    trajLine.visible = showTraj;
    groundTrack.visible = showTraj;
  });

  window.addEventListener("resize", () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  // keyboard
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") {
      e.preventDefault();
      elPlay.click();
    } else if (e.code === "ArrowRight") {
      setFrame(idx + 1);
    } else if (e.code === "ArrowLeft") {
      setFrame(idx - 1);
    }
  });

  setFrame(0);
  // initial camera
  {
    const p = enuToScene(frames.pos_enu[0]);
    camera.position.set(p.x + 40, p.y + 25, p.z - 55);
    controls.target.copy(p);
  }

  // Programmatic API for headless recording / automation.
  window.__PARAFOIL_VIEWER = {
    ready: true,
    frameCount: n,
    duration: frames.t[n - 1] - frames.t[0],
    getIndex: () => idx,
    getTime: () => frames.t[idx],
    isPlaying: () => playing,
    setFrame,
    setSpeed: (s) => {
      speed = Number(s) || 1;
      if (elSpeed) elSpeed.value = String(speed);
    },
    setCamera: (mode) => {
      const btn = document.querySelector(`[data-cam="${mode}"]`);
      if (btn) btn.click();
      else camMode = mode;
    },
    setManualRendering: (enabled) => {
      manualRendering = Boolean(enabled);
    },
    play: () => {
      if (!playing) elPlay.click();
    },
    pause: () => {
      if (playing) elPlay.click();
    },
    seekFraction: (f) => {
      setFrame(Math.round(Math.max(0, Math.min(1, f)) * (n - 1)));
    },
    seekTime: (targetTime) => {
      const target = Math.max(frames.t[0], Math.min(frames.t[n - 1], Number(targetTime)));
      let lo = 0;
      let hi = n - 1;
      while (lo < hi) {
        const mid = Math.ceil((lo + hi) / 2);
        if (frames.t[mid] <= target) lo = mid;
        else hi = mid - 1;
      }
      setFrame(lo);
      return idx >= n - 1;
    },
    /** Advance sim-time by `simDt` seconds (honors current speed only via caller). */
    stepSimTime: (simDt) => {
      const targetT = frames.t[idx] + simDt;
      let j = idx;
      while (j < n - 1 && frames.t[j + 1] <= targetT) j++;
      setFrame(j);
      return idx >= n - 1;
    },
    renderOnce: (dt = 1 / 30) => {
      updateCamera(Math.max(0, Number(dt) || 0));
      controls.enabled = camMode === "free";
      controls.update();
      renderer.render(scene, camera);
    },
    getCameraState: () => ({
      mode: camMode,
      position: camera.position.toArray(),
      target: controls.target.toArray(),
      vehiclePosition: vehicle.position.toArray(),
      vehicleQuaternion: vehicle.quaternion.toArray(),
    }),
  };

  let last = performance.now();
  function tick(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (playing) {
      accum += dt * speed;
      const dtFrame = frames.t[Math.min(idx + 1, n - 1)] - frames.t[idx] || 0.1;
      while (accum >= dtFrame && idx < n - 1) {
        accum -= dtFrame;
        setFrame(idx + 1);
        if (idx >= n - 1) {
          playing = false;
          elPlay.textContent = "Play";
        }
      }
    }
    if (!manualRendering) {
      updateCamera(dt);
      controls.enabled = camMode === "free";
      controls.update();
      renderer.render(scene, camera);
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
})();
