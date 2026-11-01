(() => {
    "use strict";

    const CLASES = ["NORM", "MI", "STTC", "CD", "HYP"];
    const COL = ["#3fa66b", "#d9372b", "#f29b38", "#6fc3e3", "#3b62c9"];
    const SPLITS = ["train", "val", "test"];
    const SPLIT_TXT = ["entrenamiento", "validación", "test"];
    const DERIV = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];
    const MI = 1;
    const MIN_N = 15;   // grupos más pequeños se marcan como indicativos
    const SUB_TXT = { AMI: "Anterior (AMI)", IMI: "Inferior (IMI)", LMI: "Lateral (LMI)", PMI: "Posterior (PMI)" };
    const ETAPA_TXT = {
        "Stadium I": "Etapa I", "Stadium II": "Etapa II", "Stadium II-III": "Etapa II–III",
        "Stadium III": "Etapa III", unknown: "Etapa desconocida", "sin dato": "Sin dato"
    };

    const $ = (s) => document.querySelector(s);
    const fmt = d3.format(",");
    const pct = d3.format(".1%");
    const pct0 = d3.format(".0%");

    // ------------------------------------------------------------ estado
    const E = {
        split: "test", sexo: "all", emin: 0, emax: 90, clases: new Set(), front: false,
        celda: null, grupo: null, varE: "sub", modoLazo: false,
        color: "label", proy: "umap", sel: -1, vecIds: [], lead: 0, vecEcg: "", alinear: false
    };
    const KMCOL = ["#6d28d9", "#db2777", "#78350f", "#0f766e", "#111827"];   // grupos de K-means (no son diagnósticos)

    let P = null, ID2I = new Map(), COOC = null;
    let vis, base, lazo = null, lazoN = 0, visList = new Int32Array(0), colIdx;
    let paleta = COL, etiquetasPaleta = CLASES;
    let bx, by, qt = null, qtCoh = null, nBase = 0;

    // ------------------------------------------------------------ carga
    async function cargarPuntos(proy) {
        const r = await fetch(`/api/puntos?proy=${proy}`);
        if (!r.ok) throw new Error(await r.text());
        const d = await r.json();
        P = {
            n: d.n, id: Int32Array.from(d.id), x: Float32Array.from(d.x), y: Float32Array.from(d.y),
            label: Uint8Array.from(d.label), pred: Uint8Array.from(d.pred), conf: Float32Array.from(d.conf),
            knn: Float32Array.from(d.knn), age: Float32Array.from(d.age), sex: Uint8Array.from(d.sex),
            split: Uint8Array.from(d.split), sub: Int8Array.from(d.sub), subs: d.subs,
            etapa: Int8Array.from(d.etapa), etapas: d.etapas, ruido: Uint8Array.from(d.ruido),
        };
        if (d.dvec) {   // datos de interpretabilidad (opcionales)
            P.dvec = Float32Array.from(d.dvec);
            P.km = Int8Array.from(d.km);
            P.kcos = Uint8Array.from(d.kcos.map((v) => (v < 0 ? 0 : v)));
            P.kmDom = d.km_dom; P.kmTam = d.km_tam;
        }
        ID2I = new Map();
        for (let i = 0; i < P.n; i++) ID2I.set(P.id[i], i);
        vis = new Uint8Array(P.n);
        base = new Uint8Array(P.n);
        colIdx = new Uint8Array(P.n);
    }

    async function cargarCooc() {
        try {
            const r = await fetch("/api/coocurrencia");
            COOC = r.ok ? await r.json() : null;
        } catch (e) { COOC = null; }
    }

    // ------------------------------------------------------------ subgrupos de MI (vista E)
    const grupoEdad = (a) => (a < 50 ? 0 : a < 65 ? 1 : a < 80 ? 2 : 3);
    const VARS = {
        sub: { n: () => P.subs.length, etq: (g) => SUB_TXT[P.subs[g]] || P.subs[g], cat: (i) => P.sub[i] },
        sexo: { n: () => 2, etq: (g) => ["Hombre", "Mujer"][g], cat: (i) => P.sex[i] },
        edad: { n: () => 4, etq: (g) => ["< 50 años", "50–64", "65–79", "80 o más"][g], cat: (i) => grupoEdad(P.age[i]) },
        ruido: { n: () => 2, etq: (g) => ["Sin ruido", "Con ruido"][g], cat: (i) => P.ruido[i] },
        etapa: { n: () => P.etapas.length, etq: (g) => ETAPA_TXT[P.etapas[g]] || P.etapas[g], cat: (i) => P.etapa[i] },
    };
    const esDelGrupo = (i) => P.label[i] === MI && VARS[E.grupo.v].cat(i) === E.grupo.g;

    // ------------------------------------------------------------ filtros
    function filtrosCohorte() {
        return { sp: E.split === "all" ? -1 : SPLITS.indexOf(E.split), sxv: E.sexo === "all" ? -1 : +E.sexo };
    }
    const pasaCohorte = (i, f) => (f.sp < 0 || P.split[i] === f.sp) && (f.sxv < 0 || P.sex[i] === f.sxv) &&
        P.age[i] >= E.emin && P.age[i] <= E.emax;

    function recomputar() {
        const f = filtrosCohorte();
        const conteo = [0, 0, 0, 0, 0];
        const cm = Array.from({ length: 5 }, () => [0, 0, 0, 0, 0]);
        const lista = [];
        nBase = 0;
        for (let i = 0; i < P.n; i++) {
            const ok = pasaCohorte(i, f) && (!lazo || lazo[i] === 1);
            base[i] = ok ? 1 : 0;
            if (!ok) { vis[i] = 0; continue; }
            nBase++;
            conteo[P.label[i]]++;
            cm[P.label[i]][P.pred[i]]++;
            let v = (E.clases.size === 0 || E.clases.has(P.label[i])) &&
                (!E.front || (P.label[i] === MI && P.pred[i] !== MI));
            if (v && E.celda) v = P.label[i] === E.celda[0] && P.pred[i] === E.celda[1];
            if (v && E.grupo) v = esDelGrupo(i);
            vis[i] = v ? 1 : 0;
            if (v) lista.push(i);
        }
        visList = Int32Array.from(lista);
        if (bx) {
            qt = d3.quadtree().x((i) => bx[i]).y((i) => by[i]).addAll(Array.from(visList));
            // «capas»: solo se pueden elegir registros del conjunto activo (Train, Val o Test)
            const coh = []; for (let i = 0; i < P.n; i++) if (base[i]) coh.push(i);
            qtCoh = d3.quadtree().x((i) => bx[i]).y((i) => by[i]).addAll(coh);
        }
        return { conteo, cm };
    }

    function actualizarColor() {
        if (E.color === "ambig") {
            // escala violeta (sin verdes, rojos, naranjas ni azules): no se confunde con los colores de los diagnósticos.
            // Poco acuerdo = violeta oscuro (resalta, es lo que se audita); mucho acuerdo = violeta muy claro.
            paleta = ["#2e1065", "#5b21b6", "#8b5cf6", "#b9a3f5", "#dcd0fa"];
            etiquetasPaleta = ["0–20 %", "20–40 %", "40–60 %", "60–80 %", "80–100 %"].map((s) => s + " de vecinos coinciden");
            for (let i = 0; i < P.n; i++) colIdx[i] = Math.min(4, Math.floor(P.knn[i] * 5));
        } else if (E.color === "knn" && P.kcos) {
            paleta = COL;
            etiquetasPaleta = CLASES;
            colIdx.set(P.kcos);
        } else if (E.color === "km" && P.km) {
            paleta = KMCOL;
            etiquetasPaleta = P.kmDom.map((c, k) => `Grupo ${k + 1} (dominado por ${c})`);
            for (let i = 0; i < P.n; i++) colIdx[i] = Math.max(0, P.km[i]);
        } else {
            paleta = COL;
            etiquetasPaleta = CLASES;
            colIdx.set(E.color === "label" ? P.label : P.pred);
        }
        const titulo = E.color === "ambig" ? "Color = acuerdo con vecinos (no es el diagnóstico):"
            : E.color === "pred" ? "Color = predicción del modelo:"
                : E.color === "knn" ? "Color = clase que votan los 10 vecinos (kNN, coseno):"
                    : E.color === "km" ? "Color = grupo de K-means (no es el diagnóstico):" : "Color = diagnóstico real:";
        $("#leyenda-meso").innerHTML = `<span><b>${titulo}</b></span>` + paleta
            .map((c, k) => `<span><i class="dot" style="background:${c}"></i>${etiquetasPaleta[k]}</span>`).join("");
    }

    function actualizarLimpiar() {
        $("#btn-limpiar").disabled = !(lazo || E.celda || E.grupo || E.clases.size || E.front || E.sel >= 0);
    }

    function refrescar() {
        const ag = recomputar();
        actualizarColor();
        pintarDonut(ag.conteo);
        pintarConfusion(ag.cm);
        pintarErrores();
        pintarStats();
        pintarAviso();
        $("#f-front").checked = E.front;
        actualizarLimpiar();
        actualizarSig();
        pedirDibujo();
    }

    // ------------------------------------------------------------ A: donut y estadísticas
    function pintarDonut(conteo) {
        const total = d3.sum(conteo);
        const S = 190, R = 90, r = 60;
        const svg = d3.select("#donut").selectAll("svg").data([0]).join("svg").attr("width", S).attr("height", S);
        const g = svg.selectAll("g").data([0]).join("g").attr("transform", `translate(${S / 2},${S / 2})`);
        const arcos = d3.pie().sort(null).value((d) => d)(conteo);
        const arc = d3.arc().innerRadius(r).outerRadius(R).padAngle(0.01);
        g.selectAll("path").data(arcos).join("path")
            .attr("d", arc).attr("fill", (_, k) => COL[k])
            .attr("opacity", (_, k) => (E.clases.size && !E.clases.has(k) ? 0.25 : 1))
            .style("cursor", "pointer").on("click", (_, d) => alternarClase(d.index));
        g.selectAll("text.t").data([0]).join("text").attr("class", "t").attr("text-anchor", "middle").attr("dy", "-0.1em")
            .style("font-size", "20px").style("font-weight", 700).text(fmt(total));
        g.selectAll("text.s").data([0]).join("text").attr("class", "s").attr("text-anchor", "middle").attr("dy", "1.4em")
            .style("font-size", "10px").style("fill", "#6b7280").text("registros");

        $("#leyenda-donut").innerHTML = CLASES.map((c, k) =>
            `<li data-k="${k}" class="${E.clases.size && !E.clases.has(k) ? "off" : ""}">
         <i class="dot" style="background:${COL[k]}"></i>${c}
         <span class="n">${fmt(conteo[k])} · ${total ? pct(conteo[k] / total) : "–"}</span></li>`).join("");
    }

    // Una sola selección a la vez (clases, celda de la matriz, subgrupo o «solo MI mal clasificados»):
    // elegir una nueva reemplaza a la anterior, para que nunca se bloqueen entre sí.
    function soloEsta(tipo) {
        if (tipo !== "clases") E.clases.clear();
        if (tipo !== "celda") E.celda = null;
        if (tipo !== "grupo") E.grupo = null;
        if (tipo !== "front") E.front = false;
    }

    function alternarClase(k) {
        soloEsta("clases");
        E.clases.has(k) ? E.clases.delete(k) : E.clases.add(k);
        refrescar();
    }

    function pintarStats() {
        let n = 0, ok = 0, mi = 0, miOk = 0, miOtra = 0;
        for (const i of visList) {
            n++;
            if (P.pred[i] === P.label[i]) ok++;
            if (P.label[i] === MI) { mi++; P.pred[i] === MI ? miOk++ : miOtra++; }
        }
        const cab = E.celda && n ? `<b>${fmt(n)} registros: real ${CLASES[E.celda[0]]}, predicho ${CLASES[E.celda[1]]}.</b><br>` : "";
        $("#stats").innerHTML = n
            ? cab + `Exactitud del modelo en lo que ves en el mapa: <b>${pct(ok / n)}</b><br>` +
            (mi ? `Recall de MI: <b>${pct(miOk / mi)}</b> (${fmt(miOk)} de ${fmt(mi)}; ${fmt(miOtra)} clasificados como otra clase)` : "Sin MI en el mapa.")
            : "Sin registros con estos filtros.";
    }

    function pintarAviso() {
        const msgs = {
            test: "Fold 10 (test): el modelo no vio estos registros durante el entrenamiento, así que la separación es real.",
            val: "Fold 9 (validación): no se usó para ajustar los pesos, pero sí para elegir el mejor modelo.",
            train: "Atención: el modelo se entrenó con estos registros; la separación puede estar inflada.",
            all: "Se mezclan entrenamiento, validación y test; filtra por Test para una lectura honesta.",
        };
        let t = msgs[E.split];
        if (lazo) t += ` Lazo activo: ${fmt(lazoN)} registros.`;
        const el = $("#aviso");
        el.textContent = t;
        el.classList.toggle("warn", E.split === "train" || E.split === "all");
    }

    // ------------------------------------------------------------ B: matriz de coocurrencia
    function pintarCooc() {
        const t = $("#cooc"), nota = $("#cooc-nota");
        if (!COOC) {
            t.innerHTML = "";
            nota.textContent = "No encuentro data/cooccurrence.json (lo genera el notebook v3.1). Cópialo a la carpeta data/.";
            return;
        }
        const cl = COOC.clases, M = COOC.matriz, K = cl.length;
        let maxOff = 1;
        for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) if (i !== j) maxOff = Math.max(maxOff, M[i][j]);
        let h = "<tr><th></th>" + cl.map((c) => `<th>${c}</th>`).join("") + "</tr>";
        for (let i = 0; i < K; i++) {
            h += `<tr><th>${cl[i]}</th>`;
            for (let j = 0; j < K; j++) {
                const v = M[i][j];
                if (i === j) {
                    h += `<td class="diag" title="Total de ${cl[i]}: ${fmt(v)} registros">${fmt(v)}</td>`;
                } else {
                    const f = Math.sqrt(v / maxOff);
                    h += `<td style="background:${d3.interpolateOranges(0.05 + 0.85 * f)};color:${f > 0.55 ? "#fff" : "#1f2937"}" ` +
                        `title="${cl[i]} y ${cl[j]}: ${fmt(v)} registros (${pct(v / M[i][i])} de ${cl[i]})">${fmt(v)}</td>`;
                }
            }
            h += "</tr>";
        }
        t.innerHTML = h;
        const perd = (COOC.por_clase || []).map((d) => `${d.clase} ${Math.round(d["perdida_%"] ?? 0)} %`).join(" · ");
        nota.innerHTML = `Registros diagnosticados: ${fmt(COOC.una_superclase + COOC.varias_superclases)}. ` +
            `Con una sola superclase: ${fmt(COOC.una_superclase)} (los usados aquí); con varias: ${fmt(COOC.varias_superclases)}. ` +
            `Descartado por clase: ${perd}. Diagonal = total por clase.`;
    }

    // ------------------------------------------------------------ D: matriz de confusión
    function pintarConfusion(cm) {
        let h = "<tr><th></th>" + CLASES.map((c) => `<th>${c}</th>`).join("") + "<th>Recall</th></tr>";
        for (let r = 0; r < 5; r++) {
            const tot = d3.sum(cm[r]);
            h += `<tr><th>${CLASES[r]}</th>`;
            for (let p = 0; p < 5; p++) {
                const f = tot ? cm[r][p] / tot : 0;
                const sel = E.celda && E.celda[0] === r && E.celda[1] === p;
                h += `<td data-r="${r}" data-p="${p}" class="${sel ? "sel" : ""}" ` +
                    `style="background:${d3.interpolateBlues(0.06 + 0.8 * f)};color:${f > 0.5 ? "#fff" : "#1f2937"}" ` +
                    `title="Real ${CLASES[r]} → predicho ${CLASES[p]}: ${fmt(cm[r][p])} (${pct(f)} de ${CLASES[r]})">${fmt(cm[r][p])}</td>`;
            }
            h += `<td class="rc">${tot ? pct0(cm[r][r] / tot) : "–"}</td></tr>`;
        }
        $("#cm").innerHTML = h;
    }

    // ------------------------------------------------------------ E: errores de MI por subgrupo
    function pintarErrores() {
        const V = VARS[E.varE], k = V.n();
        const n = new Array(k).fill(0), h = new Array(k).fill(0);
        let nt = 0, ht = 0;
        for (let i = 0; i < P.n; i++) {
            if (!base[i] || P.label[i] !== MI) continue;
            nt++;
            const hit = P.pred[i] === MI;
            if (hit) ht++;
            const g = V.cat(i);
            if (g >= 0) { n[g]++; if (hit) h[g]++; }
        }
        const cont = $("#err");
        if (!nt) { cont.innerHTML = '<p class="nota">No hay infartos en la selección.</p>'; return; }
        const ro = ht / nt;
        let html = `<p class="er-tit">Recall de MI en la selección: <b>${pct(ro)}</b> (${fmt(ht)} de ${fmt(nt)}). ` +
            `La línea punteada marca ese valor.</p>`;
        for (let g = 0; g < k; g++) {
            if (!n[g]) continue;
            const r = h[g] / n[g];
            const on = E.grupo && E.grupo.v === E.varE && E.grupo.g === g;
            html += `<div class="er ${on ? "on" : ""} ${n[g] < MIN_N ? "peq" : ""}" data-g="${g}" ` +
                `title="${n[g] < MIN_N ? "Grupo pequeño (n < 15): solo indicativo. " : ""}Clic para resaltar este subgrupo en el mapa">` +
                `<span>${V.etq(g)}</span><div class="eb"><i style="width:${(r * 100).toFixed(1)}%"></i>` +
                `<s style="left:${(ro * 100).toFixed(1)}%"></s></div><b>${pct0(r)}</b><small>n=${n[g]}</small></div>`;
        }
        cont.innerHTML = html;
    }

    // ------------------------------------------------------------ C: dispersión en canvas
    const wrap = $("#scatter-wrap"), canvas = $("#scatter"), ctx = canvas.getContext("2d");
    const tip = $("#tip");
    let W = 0, H = 0, dpr = 1, tf = d3.zoomIdentity, polig = null;

    function ajustar() {
        const r = wrap.getBoundingClientRect();
        W = r.width; H = r.height; dpr = window.devicePixelRatio || 1;
        canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
        if (!P || W < 10 || H < 10) return;
        const m = 18;
        const sxs = d3.scaleLinear().domain(d3.extent(P.x)).range([m, W - m]);
        const sys = d3.scaleLinear().domain(d3.extent(P.y)).range([H - m, m]);
        bx = new Float32Array(P.n); by = new Float32Array(P.n);
        for (let i = 0; i < P.n; i++) { bx[i] = sxs(P.x[i]); by[i] = sys(P.y[i]); }
        refrescar();
    }

    let pendiente = false;
    function pedirDibujo() {
        if (pendiente) return;
        pendiente = true;
        requestAnimationFrame(() => { pendiente = false; dibujar(); });
    }

    const vecinosIdx = () => E.vecIds.map((id) => ID2I.get(id)).filter((v) => v !== undefined);

    function dibujar() {
        if (!P || !bx) return;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, W, H);
        const s = Math.min(7, 2.4 * Math.pow(tf.k, 0.45)), h = s / 2;

        // registros fuera del filtro: tenues, solo para dar contexto
        const filtrado = visList.length < nBase;
        ctx.globalAlpha = 0.5;
        ctx.fillStyle = "#d4d9e1";
        for (let i = 0; i < P.n; i++) {
            if (vis[i] || !base[i]) continue;
            const x = tf.applyX(bx[i]), y = tf.applyY(by[i]);
            if (x < -s || x > W + s || y < -s || y > H + s) continue;
            ctx.fillRect(x - h, y - h, s, s);
        }
        // registros visibles, agrupados por color (más grandes si hay una selección activa, para que resalten)
        const sv = filtrado ? s + 1.5 : s, hv = sv / 2;
        ctx.globalAlpha = filtrado ? 0.95 : 0.8;
        for (let c = 0; c < paleta.length; c++) {
            ctx.fillStyle = paleta[c];
            for (const i of visList) {
                if (colIdx[i] !== c) continue;
                const x = tf.applyX(bx[i]), y = tf.applyY(by[i]);
                if (x < -s || x > W + s || y < -s || y > H + s) continue;
                ctx.fillRect(x - hv, y - hv, sv, sv);
            }
        }
        ctx.globalAlpha = 1;

        // paciente seleccionado y sus vecinos
        if (E.sel >= 0) {
            const sx0 = tf.applyX(bx[E.sel]), sy0 = tf.applyY(by[E.sel]);
            const vs = vecinosIdx();
            ctx.strokeStyle = "rgba(17,24,39,.35)"; ctx.lineWidth = 1;
            for (const v of vs) {
                ctx.beginPath(); ctx.moveTo(sx0, sy0); ctx.lineTo(tf.applyX(bx[v]), tf.applyY(by[v])); ctx.stroke();
            }
            for (const v of vs) {
                ctx.beginPath(); ctx.arc(tf.applyX(bx[v]), tf.applyY(by[v]), 4.5, 0, 6.2832);
                ctx.fillStyle = COL[P.label[v]]; ctx.fill();
                ctx.lineWidth = 1.5; ctx.strokeStyle = "#fff"; ctx.stroke();
            }
            const kc = E.vecEcg ? ID2I.get(+E.vecEcg) : undefined;   // ECG con el que se compara la señal
            if (kc !== undefined) {
                const cx = tf.applyX(bx[kc]), cy = tf.applyY(by[kc]);
                ctx.beginPath(); ctx.arc(cx, cy, 8, 0, 6.2832);
                ctx.lineWidth = 4; ctx.strokeStyle = "#fff"; ctx.stroke();
                ctx.lineWidth = 2; ctx.strokeStyle = "#7c3aed"; ctx.stroke();
                ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(sx0, sy0); ctx.lineTo(cx, cy);
                ctx.lineWidth = 1.5; ctx.strokeStyle = "#7c3aed"; ctx.stroke(); ctx.setLineDash([]);
            }
            ctx.beginPath(); ctx.arc(sx0, sy0, 7.5, 0, 6.2832);
            ctx.lineWidth = 4; ctx.strokeStyle = "#fff"; ctx.stroke();
            ctx.lineWidth = 2; ctx.strokeStyle = "#111827"; ctx.stroke();
            // número de orden de cada vecino (1 = el más cercano en el espacio de 64 números)
            ctx.font = "bold 10px sans-serif"; ctx.textBaseline = "middle";
            E.vecIds.forEach((id, k) => {
                const v = ID2I.get(id); if (v === undefined) return;
                const tx = tf.applyX(bx[v]) + 7, ty = tf.applyY(by[v]) - 7, t = String(k + 1);
                ctx.lineWidth = 3; ctx.strokeStyle = "#fff"; ctx.strokeText(t, tx, ty);
                ctx.fillStyle = "#111827"; ctx.fillText(t, tx, ty);
            });
        }

        // lazo en curso
        if (polig && polig.length > 1) {
            ctx.beginPath();
            ctx.moveTo(polig[0][0], polig[0][1]);
            for (let k = 1; k < polig.length; k++) ctx.lineTo(polig[k][0], polig[k][1]);
            ctx.closePath();
            ctx.fillStyle = "rgba(43,74,139,.08)"; ctx.fill();
            ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5; ctx.strokeStyle = "#2b4a8b"; ctx.stroke();
            ctx.setLineDash([]);
        }
    }

    function buscarPunto(ev) {
        if (!qt) return undefined;
        const r = canvas.getBoundingClientRect();
        // los vecinos del paciente seleccionado se pueden elegir aunque el filtro los oculte
        if (E.sel >= 0) {
            const mx = ev.clientX - r.left, my = ev.clientY - r.top;
            let mejor, md = 100;   // radio de 10 px
            for (const v of vecinosIdx()) {
                const dx = tf.applyX(bx[v]) - mx, dy = tf.applyY(by[v]) - my, dd = dx * dx + dy * dy;
                if (dd < md) { md = dd; mejor = v; }
            }
            if (mejor !== undefined) return mejor;
        }
        const px = (ev.clientX - r.left - tf.x) / tf.k, py = (ev.clientY - r.top - tf.y) / tf.k;
        const rad = 10 / tf.k;
        // con Mayús se busca entre todo el conjunto activo, no solo entre los resaltados
        if (ev.shiftKey && qtCoh) return qtCoh.find(px, py, rad);
        // primero los resaltados; si no hay ninguno cerca, los tenues del mismo conjunto
        const v = qt.find(px, py, rad);
        return v !== undefined ? v : (qtCoh ? qtCoh.find(px, py, rad) : undefined);
    }

    const punto = (ev) => { const r = canvas.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };

    function cerrarLazo() {
        const poly = polig;
        polig = null;
        if (!poly || poly.length < 4) { pedirDibujo(); return; }
        // el lazo se usa una vez: al terminar vuelve el modo normal (zoom y clic en puntos)
        E.modoLazo = false; $("#btn-lazo").classList.remove("on"); canvas.style.cursor = "crosshair";
        const f = filtrosCohorte();
        const m = new Uint8Array(P.n);
        let c = 0;
        for (let i = 0; i < P.n; i++) {
            if (!pasaCohorte(i, f)) continue;
            if (d3.polygonContains(poly, [tf.applyX(bx[i]), tf.applyY(by[i])])) { m[i] = 1; c++; }
        }
        lazo = c ? m : null;
        lazoN = c;
        E.celda = null; E.grupo = null;
        refrescar();
    }

    canvas.addEventListener("pointerdown", (ev) => {
        if (!E.modoLazo) return;
        polig = [punto(ev)];
        canvas.setPointerCapture(ev.pointerId);
        tip.style.display = "none";
    });
    canvas.addEventListener("pointermove", (ev) => {
        if (polig) { polig.push(punto(ev)); pedirDibujo(); return; }
        if (E.modoLazo) return;
        const i = buscarPunto(ev);
        if (i === undefined) { tip.style.display = "none"; canvas.style.cursor = "crosshair"; return; }
        canvas.style.cursor = "pointer";
        const r = wrap.getBoundingClientRect();
        tip.innerHTML = `<b>ECG ${P.id[i]}</b><br>Real: ${CLASES[P.label[i]]} · Predicción: ${CLASES[P.pred[i]]} (${pct(P.conf[i])})` +
            (vis[i] ? "" : `<br><i>Fuera del filtro actual (conjunto ${SPLITS[P.split[i]]})</i>`) +
            `<br>Vecinos con la misma etiqueta: ${Math.round(P.knn[i] * 10)}/10` +
            (P.dvec ? `<br>kNN (coseno): ${CLASES[P.kcos[i]]} · dist. media a vecinos: ${(P.dvec[i] * 1000).toFixed(1)}×10⁻³` : "") +
            (E.vecIds.includes(P.id[i]) ? `<br><b>Vecino ${E.vecIds.indexOf(P.id[i]) + 1} de ${P.id[E.sel]}</b>` +
                (distVec.get(P.id[i]) !== undefined ? ` · distancia ${(distVec.get(P.id[i]) * 1000).toFixed(1)}×10⁻³` : "") : "") +
            (E.sel >= 0 && i !== E.sel ? "<br><i>Clic: elegirlo · Mayús + clic: comparar la señal</i>" : "");
        tip.style.display = "block";
        const tx = ev.clientX - r.left + 12, ty = ev.clientY - r.top + 12;
        tip.style.left = Math.min(tx, r.width - tip.offsetWidth - 6) + "px";
        tip.style.top = Math.min(ty, r.height - tip.offsetHeight - 6) + "px";
    });
    canvas.addEventListener("pointerup", () => { if (polig) cerrarLazo(); });
    canvas.addEventListener("pointerleave", () => { tip.style.display = "none"; });
    canvas.addEventListener("click", (ev) => {
        if (E.modoLazo) return;
        const i = buscarPunto(ev);
        if (i === undefined) return;
        if (ev.shiftKey && E.sel >= 0) compararCon(P.id[i]);   // Mayús + clic: superponer la señal de otro ECG
        else if (i === E.sel) deseleccionar();                  // clic en el ya elegido: soltarlo
        else seleccionar(i);
    });

    const zoom = d3.zoom().scaleExtent([1, 40]).clickDistance(8)
        .filter((ev) => !E.modoLazo && (!ev.ctrlKey || ev.type === "wheel") && !ev.button)
        .on("zoom", (e) => { tf = e.transform; pedirDibujo(); });
    d3.select(canvas).call(zoom);

    // ------------------------------------------------------------ F: ficha del paciente
    async function seleccionar(i) {
        E.sel = i;
        actualizarLimpiar();
        E.vecEcg = "";
        ecgVec = null;
        $("#i-otro").value = "";
        pedirDibujo();
        const r = await fetch(`/api/paciente/${P.id[i]}`);
        if (!r.ok) return;
        const d = await r.json();
        if (E.sel !== i) return;   // el usuario ya eligió otro
        E.vecIds = d.vecinos;
        document.dispatchEvent(new CustomEvent("ptbxl:seleccion", { detail: { ecg: d.ecg_id } }));
        pintarFicha(d);
        actualizarSig();
        pedirDibujo();
        $("#vacio").hidden = true;
        $("#detalle").hidden = false;
        await cargarSenal();
    }

    function pintarFicha(d) {
        const kl = CLASES.indexOf(d.label), kp = CLASES.indexOf(d.prediccion);
        $("#f-titulo").textContent = `ECG ${d.ecg_id}`;
        const extra = d.subtipo ? ` · ${SUB_TXT[d.subtipo] || d.subtipo}` : "";
        $("#f-meta").textContent = `Paciente ${d.patient_id} · ${d.sexo}, ${d.edad === null ? ">89 años" : d.edad + " años"} · fold ${d.fold}${extra}`;
        $("#f-badges").innerHTML =
            `<span class="badge" style="background:${COL[kl]}">Real: ${d.label}</span>` +
            `<span class="badge" style="background:${COL[kp]}">Modelo: ${d.prediccion}</span>` +
            (kl !== kp ? `<span class="badge warn">Discrepancia</span>` : "");
        $("#f-probs").innerHTML = d.probs.map((p, k) =>
            `<div class="pb"><span>${CLASES[k]}</span><div class="barra"><i style="width:${(p * 100).toFixed(1)}%;background:${COL[k]}"></i></div><b>${(p * 100).toFixed(1)}%</b></div>`).join("");
        const nm = Math.round(d.frac_vecinos_MI * 10);
        let txt = `${Math.round(d.knn_acuerdo * 10)} de 10 vecinos (entrenamiento) comparten la etiqueta real; ${nm} son MI.`;
        if (d.label === "MI" && d.prediccion !== "MI")
            txt += " <b>Caso frontera:</b> ni el modelo ni sus vecinos lo reconocen como infarto.";
        $("#f-knn").innerHTML = txt;
        const dv = d.interp ? d.interp.dist_vecinos : [];
        distVec = new Map(d.vecinos.map((id, k) => [id, dv[k]]));
        const dTxt = (id) => (distVec.get(id) === undefined ? "" : (distVec.get(id) * 1000).toFixed(1));
        $("#f-vecinos").innerHTML = d.vecinos.map((id) => {
            const i = ID2I.get(id);
            return i === undefined ? "" :
                `<button class="chip" data-id="${id}" title="${distVec.get(id) === undefined ? "" : "Distancia coseno " + distVec.get(id).toFixed(5)}">` +
                `<i class="dot" style="background:${COL[P.label[i]]}"></i>${id}` +
                (dTxt(id) ? ` <small>${dTxt(id)}</small>` : "") + `</button>`;
        }).join("");
        $("#s-vec").innerHTML = '<option value="">Comparar con un vecino…</option>' + d.vecinos.map((id) => {
            const i = ID2I.get(id);
            return i === undefined ? "" : `<option value="${id}">ECG ${id} · ${CLASES[P.label[i]]}` +
                (dTxt(id) ? ` · d=${dTxt(id)}` : "") + `</option>`;
        }).join("");
        $("#s-vec").value = "";
        pintarInterp(d);
    }

    let distVec = new Map();

    // percentil de la distancia media a los vecinos dentro del mismo conjunto (train / val / test)
    function percentilDvec(i) {
        if (!P.dvec) return null;
        const sp = P.split[i], v = P.dvec[i];
        let n = 0, menor = 0;
        for (let j = 0; j < P.n; j++) if (P.split[j] === sp && P.dvec[j] >= 0) { n++; if (P.dvec[j] < v) menor++; }
        return n ? menor / n : null;
    }

    function pintarInterp(d) {
        const el = $("#f-interp"), it = d.interp;
        if (!it) { el.innerHTML = ""; return; }
        const pc = percentilDvec(E.sel);
        const lejos = pc !== null && pc >= 0.9;
        const k = it.knn;
        const distinto = k.coseno !== d.prediccion;
        // vecinos que se ven lejos en el mapa aunque estén cerca en los 64 números
        const lejanos = [];
        if (bx && E.sel >= 0) {
            const dd = [];
            d.vecinos.forEach((id, k) => {
                const v = ID2I.get(id);
                if (v !== undefined) dd.push([k + 1, Math.hypot(bx[v] - bx[E.sel], by[v] - by[E.sel])]);
            });
            const med = d3.median(dd, (x) => x[1]) || 0;
            dd.forEach(([k, px]) => { if (px > Math.max(30, 4 * med)) lejanos.push(k); });   // mucho más lejos que el resto
        }
        el.innerHTML =
            `<p><span class="k">Distancia media a sus 10 vecinos:</span> <b>${(it.dvec * 1000).toFixed(1)}</b>×10<sup>-3</sup>` +
            (pc !== null ? ` <span class="k">(más lejos que el ${Math.round(pc * 100)} % de los registros de ${d.split})</span>` : "") +
            (lejos ? ` <span class="lejos">· vecinos lejanos</span>` : "") + `</p>` +
            `<p><span class="k">Voto de los 10 vecinos:</span> <b>${k.coseno}</b> (coseno) · ${k.manhattan} (Manhattan) · ${k.chebyshev} (Chebyshev)` +
            ` <span class="k">· Modelo: ${d.prediccion}</span>` +
            (distinto ? ` <span class="lejos">· el kNN y el modelo no coinciden</span>` : "") + `</p>` +
            `<p><span class="k">Grupos:</span> K-means ${it.kmeans.grupo + 1} (dominado por ${it.kmeans.dominante}) · ` +
            `DBSCAN ${it.dbscan < 0 ? "ruido" : "grupo " + (it.dbscan + 1)}</p>` +
            (lejanos.length ? `<p class="k">El vecino ${lejanos.join(", ")} se ve lejos en el mapa, pero está cerca en los 64 números ` +
                `(su distancia está en la lista de abajo): el mapa es una proyección a 2D y deforma algunas distancias.</p>` : "");
    }

    $("#f-vecinos").addEventListener("click", (ev) => {
        const b = ev.target.closest(".chip");
        if (b) seleccionar(ID2I.get(+b.dataset.id));
    });

    // ------------------------------------------------------------ G: señal ECG
    const M = { t: 10, r: 14, b: 26, l: 46 };
    let ecg = null, ecgVec = null, x0, zx, yS, zoomE, ultimoPaciente = -1;

    async function cargarSenal() {
        const id = P.id[E.sel];
        const r = await fetch(`/api/senal/${id}?derivacion=${E.lead}`);
        if (!r.ok || P.id[E.sel] !== id) return;
        ecg = await r.json();
        await cargarVecino(false);
        if (P.id[E.sel] !== id) return;
        const nuevo = ultimoPaciente !== id;
        ultimoPaciente = id;
        montarECG(nuevo);
    }

    async function cargarVecino(redibujar) {
        ecgVec = null;
        const pedido = E.vecEcg;
        if (pedido) {
            try {
                const r = await fetch(`/api/senal/${pedido}?derivacion=${E.lead}`);
                if (r.ok && E.vecEcg === pedido) ecgVec = await r.json();
            } catch (e) { ecgVec = null; }
        }
        pintarLeyendaECG();
        if (redibujar) montarECG(false);
    }

    function pintarLeyendaECG() {
        const el = $("#ecg-ley");
        if (E.sel < 0 || !ecg) { el.innerHTML = ""; return; }
        const desc = (i) => `ECG ${P.id[i]} · real ${CLASES[P.label[i]]}, modelo ${CLASES[P.pred[i]]}`;
        let h = `<span><i style="border-color:#1f2937"></i>${desc(E.sel)}</span>`;
        const k = E.vecEcg ? ID2I.get(+E.vecEcg) : undefined;
        if (ecgVec && k !== undefined) {
            const dd = distVec.get(+E.vecEcg);
            h += `<span><i style="border-color:#7c3aed"></i>${desc(k)}` +
                (dd !== undefined ? ` · distancia ${(dd * 1000).toFixed(1)}×10<sup>-3</sup>` : "") + `</span>`;
        }
        el.innerHTML = h;
    }

    // superpone la señal de cualquier ECG a la del paciente seleccionado
    function compararCon(id) {
        if (E.sel < 0) return;
        if (!ID2I.has(id)) { $("#aviso").textContent = `El ECG ${id} no está en el conjunto.`; return; }
        E.vecEcg = id === P.id[E.sel] ? "" : String(id);
        const sv = $("#s-vec");
        sv.querySelectorAll("option[data-extra]").forEach((o) => o.remove());
        if (E.vecEcg && !Array.from(sv.options).some((o) => o.value === E.vecEcg)) {
            const o = document.createElement("option");
            o.value = E.vecEcg; o.dataset.extra = "1";
            o.textContent = `ECG ${id} · ${CLASES[P.label[ID2I.get(id)]]} (elegido)`;
            sv.appendChild(o);
        }
        sv.value = E.vecEcg;
        $("#i-otro").value = E.vecEcg && !distVec.has(id) ? E.vecEcg : "";
        pedirDibujo();
        return cargarVecino(true);
    }

    // posición aproximada del complejo QRS más marcado en los primeros 2 s (mayor pendiente)
    function posQRS(mv) {
        let b = 1, m = 0;
        const n = Math.min(mv.length - 1, 200);
        for (let i = 1; i < n; i++) { const v = Math.abs(mv[i] - mv[i - 1]); if (v > m) { m = v; b = i; } }
        return b;
    }

    // pasa al siguiente caso de la selección actual (celda de la matriz, subgrupo, lazo...)
    function ordenSeleccion() { return Array.from(visList).sort((a, b) => P.id[a] - P.id[b]); }
    function siguienteCaso() {
        if (!visList.length) return;
        const o = ordenSeleccion(), k = o.indexOf(E.sel);
        seleccionar(o[(k + 1) % o.length]);
    }
    function actualizarSig() {
        const b = $("#btn-sig"); if (!b || !P) return;
        const o = ordenSeleccion(), k = o.indexOf(E.sel);
        b.textContent = `Siguiente caso (${k >= 0 ? (k + 1) + "/" : ""}${o.length})`;
        b.disabled = !o.length;
    }

    function montarECG(reset) {
        if (!ecg) return;
        const r = $("#ecg-wrap").getBoundingClientRect();
        const w = r.width, h = r.height;
        if (w < 10 || h < 10) return;
        const svg = d3.select("#ecg").attr("width", w).attr("height", h);
        svg.selectAll("*").remove();

        const T = ecg.mv.length / ecg.fs;
        x0 = d3.scaleLinear().domain([0, T]).range([M.l, w - M.r]);
        const todos = ecgVec ? ecg.mv.concat(ecgVec.mv) : ecg.mv;
        const ext = d3.extent(todos), pad = (ext[1] - ext[0]) * 0.1 || 0.1;
        yS = d3.scaleLinear().domain([ext[0] - pad, ext[1] + pad]).range([h - M.b, M.t]);

        svg.append("clipPath").attr("id", "clip").append("rect")
            .attr("x", M.l).attr("y", 0).attr("width", w - M.l - M.r).attr("height", h);
        svg.append("g").attr("class", "grid");
        const g = svg.append("g").attr("clip-path", "url(#clip)");
        g.append("path").attr("class", "trazo-vec");
        g.append("path").attr("class", "trazo");
        svg.append("g").attr("class", "ejex").attr("transform", `translate(0,${h - M.b})`);
        svg.append("g").attr("class", "ejey").attr("transform", `translate(${M.l},0)`);
        svg.append("text").attr("x", 4).attr("y", M.t + 4).style("font-size", "10px").style("fill", "#6b7280")
            .text(`mV · ${ecg.derivacion}`);

        zoomE = d3.zoom().scaleExtent([1, 10])
            .translateExtent([[M.l, 0], [w - M.r, h]]).extent([[M.l, 0], [w - M.r, h]])
            .on("zoom", (e) => { zx = e.transform.rescaleX(x0); renderECG(); });
        svg.call(zoomE);
        if (reset) svg.call(zoomE.transform, d3.zoomIdentity);   // dispara renderECG
        else { zx = d3.zoomTransform(svg.node()).rescaleX(x0); renderECG(); }
    }

    function renderECG() {
        if (!ecg || !zx) return;
        const svg = d3.select("#ecg");
        const w = +svg.attr("width"), h = +svg.attr("height");
        svg.select(".grid").selectAll("line.v").data(zx.ticks(10)).join("line").attr("class", "v")
            .attr("x1", (d) => zx(d)).attr("x2", (d) => zx(d)).attr("y1", M.t).attr("y2", h - M.b);
        svg.select(".grid").selectAll("line.h").data(yS.ticks(6)).join("line").attr("class", "h")
            .attr("x1", M.l).attr("x2", w - M.r).attr("y1", (d) => yS(d)).attr("y2", (d) => yS(d));
        const fs = ecg.fs;
        const linea = d3.line().x((_, i) => zx(i / fs)).y((d) => yS(d));
        svg.select(".trazo").attr("d", linea(ecg.mv));
        const sh = E.alinear && ecgVec ? posQRS(ecg.mv) - posQRS(ecgVec.mv) : 0;   // desfase para alinear el QRS
        const lineaV = d3.line().x((_, i) => zx((i + sh) / fs)).y((d) => yS(d));
        svg.select(".trazo-vec").attr("d", ecgVec ? lineaV(ecgVec.mv) : null);
        svg.select(".ejex").call(d3.axisBottom(zx).ticks(10).tickFormat((d) => d + " s"));
        svg.select(".ejey").call(d3.axisLeft(yS).ticks(6));
    }

    // ------------------------------------------------------------ controles
    function enlazar() {
        $("#f-split").addEventListener("click", (ev) => {
            const b = ev.target.closest("button"); if (!b) return;
            E.split = b.dataset.v;
            document.querySelectorAll("#f-split button").forEach((x) => x.classList.toggle("on", x === b));
            lazo = null;   // el lazo se define sobre una cohorte concreta
            refrescar();
        });
        $("#f-sexo").addEventListener("change", (ev) => { E.sexo = ev.target.value; lazo = null; refrescar(); });
        const edades = () => {
            let a = +$("#f-emin").value, b = +$("#f-emax").value;
            if (a > b) { [a, b] = [b, a]; }
            E.emin = a; E.emax = b;
            $("#v-edad").textContent = `${a}–${b}${b === 90 ? "+" : ""}`;
            lazo = null;
            refrescar();
        };
        $("#f-emin").addEventListener("input", edades);
        $("#f-emax").addEventListener("input", edades);
        $("#f-front").addEventListener("change", (ev) => { if (ev.target.checked) soloEsta("front"); E.front = ev.target.checked; refrescar(); });
        $("#leyenda-donut").addEventListener("click", (ev) => {
            const li = ev.target.closest("li"); if (li) alternarClase(+li.dataset.k);
        });
        $("#cm").addEventListener("click", (ev) => {
            const td = ev.target.closest("td[data-r]"); if (!td) return;
            const r = +td.dataset.r, p = +td.dataset.p;
            const misma = E.celda && E.celda[0] === r && E.celda[1] === p;
            soloEsta("celda");
            E.celda = misma ? null : [r, p];
            refrescar();
        });
        $("#s-var").addEventListener("change", (ev) => { E.varE = ev.target.value; E.grupo = null; refrescar(); });
        document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && !document.querySelector("dialog[open]")) limpiarTodo(); });
        $("#err").addEventListener("click", (ev) => {
            const row = ev.target.closest(".er"); if (!row) return;
            const g = +row.dataset.g;
            const mismo = E.grupo && E.grupo.v === E.varE && E.grupo.g === g;
            soloEsta("grupo");
            E.grupo = mismo ? null : { v: E.varE, g };
            refrescar();
        });

        $("#s-color").addEventListener("change", (ev) => { E.color = ev.target.value; actualizarColor(); pedirDibujo(); });
        $("#s-proy").addEventListener("change", async (ev) => {
            const idSel = E.sel >= 0 ? P.id[E.sel] : null;
            E.proy = ev.target.value;
            try { await cargarPuntos(E.proy); }
            catch (e) { $("#aviso").textContent = "No se pudo cargar la proyección: " + e.message; return; }
            E.sel = idSel !== null && ID2I.has(idSel) ? ID2I.get(idSel) : -1;
            tf = d3.zoomIdentity;
            d3.select(canvas).call(zoom.transform, d3.zoomIdentity);
            ajustar();
        });
        $("#btn-reset").addEventListener("click", () => {
            d3.select(canvas).transition().duration(350).call(zoom.transform, d3.zoomIdentity);
        });
        $("#btn-lazo").addEventListener("click", () => {
            E.modoLazo = !E.modoLazo;
            $("#btn-lazo").classList.toggle("on", E.modoLazo);
            canvas.style.cursor = "crosshair";
        });
        $("#btn-limpiar").addEventListener("click", limpiarTodo);

        const sl = $("#s-lead");
        sl.innerHTML = DERIV.map((d, k) => `<option value="${k}">Derivación ${d}</option>`).join("");
        sl.value = E.lead;
        sl.addEventListener("change", (ev) => { E.lead = +ev.target.value; if (E.sel >= 0) cargarSenal(); });
        $("#s-vec").addEventListener("change", (ev) => {
            if (!ev.target.value) { E.vecEcg = ""; $("#i-otro").value = ""; pedirDibujo(); cargarVecino(true); }
            else compararCon(+ev.target.value);
        });
        const otro = () => { const v = +$("#i-otro").value; if (v) compararCon(v); };
        $("#i-otro").addEventListener("keydown", (ev) => { if (ev.key === "Enter") otro(); });
        $("#i-otro").addEventListener("change", otro);
        $("#c-alinear").addEventListener("change", (ev) => { E.alinear = ev.target.checked; renderECG(); });
        $("#btn-sig").addEventListener("click", siguienteCaso);

        $("#btn-buscar").addEventListener("click", buscar);
        $("#buscar").addEventListener("keydown", (ev) => { if (ev.key === "Enter") buscar(); });
        $("#btn-frontera").addEventListener("click", () => {
            const cand = [];
            for (const i of visList) if (P.label[i] === MI && P.pred[i] !== MI) cand.push(i);
            if (!cand.length) { $("#aviso").textContent = "No hay MI mal clasificados en la cohorte actual."; return; }
            seleccionar(cand[Math.floor(Math.random() * cand.length)]);
        });
    }

    // quita el paciente elegido: vuelve a la vista sin selección (la ficha y la señal se vacían)
    function deseleccionar() {
        E.sel = -1; E.vecIds = []; E.vecEcg = ""; ecg = null; ecgVec = null; ultimoPaciente = -1;
        $("#vacio").hidden = false;
        $("#detalle").hidden = true;
        document.dispatchEvent(new CustomEvent("ptbxl:seleccion", { detail: { ecg: null } }));
        actualizarLimpiar();
        actualizarSig();
        pedirDibujo();
    }

    // «Limpiar selección»: deja el mapa como al empezar (sin lazo, sin filtros de resaltado y sin paciente elegido)
    function limpiarTodo() {
        lazo = null; lazoN = 0; E.celda = null; E.grupo = null; E.clases.clear(); E.front = false;
        if (E.sel >= 0) deseleccionar();
        refrescar();
    }

    function buscar() {
        const id = +$("#buscar").value;
        const i = ID2I.get(id);
        if (i === undefined) { $("#aviso").textContent = `El ECG ${id} no está en el conjunto (solo hay registros con una superclase).`; return; }
        if (!base[i]) {
            const sp = SPLIT_TXT[P.split[i]];
            $("#aviso").textContent = `El ECG ${id} pertenece a ${sp}. Cambia el conjunto de datos (o los filtros de sexo y edad) para poder elegirlo.`;
            return;
        }
        seleccionar(i);
    }

    // ------------------------------------------------------------ inicio
    (async function init() {
        try {
            await Promise.all([cargarPuntos(E.proy), cargarCooc()]);
        } catch (e) {
            $("#aviso").textContent = "No se pudieron cargar los datos: " + e.message;
            return;
        }
        if (!P.km) document.querySelectorAll('#s-color option[value="knn"], #s-color option[value="km"]').forEach((o) => o.remove());
        enlazar();
        pintarCooc();
        new ResizeObserver(() => ajustar()).observe(wrap);
        new ResizeObserver(() => montarECG(false)).observe($("#ecg-wrap"));
        ajustar();
    })();
})();