/* Comparación de un ECG con sus 10 vecinos (ventana emergente). Usa d3 (ya cargado). */
(() => {
    "use strict";
    const DERIV = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];
    const COL = { NORM: "#3fa66b", MI: "#d9372b", STTC: "#f29b38", CD: "#6fc3e3", HYP: "#3b62c9" };
    const NOM = { NORM: "Normal", MI: "Infarto", STTC: "ST-T", CD: "Conducción", HYP: "Hipertrofia" };
    const NB_COL = ["#16a34a", "#0d9488", "#92400e", "#111827", "#84cc16", "#c026d3", "#ca8a04", "#6b7280"];
    const NB_DASH = ["6 3", "2 3", "9 3 2 3"];
    const colNb = (k) => NB_COL[(k - 1) % NB_COL.length];
    const dashNb = (k) => NB_DASH[Math.floor((k - 1) / NB_COL.length) % NB_DASH.length];
    const UMBRAL = 0.2;
    const $ = (s) => document.querySelector(s);
    const cache = new Map();
    const E = { ecg: null, datos: null, dist: [], lead: 1, ref: "todos", sel: [], cargando: false };

    document.addEventListener("ptbxl:seleccion", (ev) => { E.ecg = ev.detail.ecg; $("#btn-comparar").disabled = !ev.detail.ecg; });

    const med = (v) => { const a = v.filter((x) => x !== null && x !== undefined).sort((p, q) => p - q); const n = a.length;
        return n ? (n % 2 ? a[(n - 1) / 2] : (a[n / 2 - 1] + a[n / 2]) / 2) : 0; };
    const ptp = (a) => d3.max(a) - d3.min(a);
    const corr = (a, b) => { const ma = d3.mean(a), mb = d3.mean(b); let sab = 0, saa = 0, sbb = 0;
        for (let i = 0; i < a.length; i++) { const x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
        return sab / Math.sqrt(saa * sbb || 1); };
    const plano = (beat) => beat.flat();

    async function abrir() {
        if (E.ecg === null || E.cargando) return;
        E.cargando = true;
        const dlg = $("#dlg-comp");
        $("#comp-cuerpo").innerHTML = "<p>Calculando los latidos medios…</p>";
        if (!dlg.open) dlg.showModal();
        try {
            if (!cache.has(E.ecg)) {
                const [a, b] = await Promise.all([fetch(`/api/comparar/${E.ecg}`), fetch(`/api/paciente/${E.ecg}`)]);
                if (!a.ok) throw new Error(await a.text());
                const comp = await a.json(), pac = b.ok ? await b.json() : null;
                cache.set(E.ecg, { comp, dist: pac && pac.interp ? pac.interp.dist_vecinos : [], pac });
            }
            const c = cache.get(E.ecg);
            E.datos = c.comp; E.dist = c.dist; E.pac = c.pac;
            E.primera = true;
            E.sel = []; E.ref = "todos";
            armar();
        } catch (e) {
            $("#comp-cuerpo").innerHTML = `<p>No se pudo calcular: ${e.message}</p>`;
        } finally { E.cargando = false; }
    }

    function plantilla() {
        const R = E.datos.registros, caso = R[0], vec = R.slice(1).filter((r) => r.beat);
        let sub = vec, nota = "todos los vecinos";
        if (E.ref === "otra") { sub = vec.filter((r) => r.label !== caso.label); nota = "los vecinos de otra clase"; }
        if (E.ref === "misma") { sub = vec.filter((r) => r.label === caso.label); nota = "los vecinos de su misma clase"; }
        let aviso = false;
        if (!sub.length) { sub = vec; nota = "todos los vecinos (no hay de ese tipo)"; aviso = true; }
        const T = DERIV.map((_, l) => caso.beat[0].map((_, t) => med(sub.map((r) => r.beat[l][t]))));
        return { T, n: sub.length, nota, ks: new Set(sub.map((r) => r.k)), aviso };
    }

    function difs(caso, T) {
        return DERIV.map((_, l) => {
            const esc = Math.max(0.2, ptp(T[l]));
            const nd = caso.beat[l].map((v, t) => Math.abs(v - T[l][t]) / esc);
            return { nd, score: d3.mean(nd) };
        });
    }

    const ms = (t) => (t - E.datos.pre) * 10;   // 100 Hz = 10 ms por muestra
    function tramo(m) { return m < -40 ? "antes del QRS (P / línea base)" : m <= 60 ? "QRS" : m <= 160 ? "segmento ST" : "onda T"; }

    function armar() {
        const D = E.datos, caso = D.registros[0];
        if (!caso.beat) { $("#comp-cuerpo").innerHTML = "<p>No se pudo detectar un latido estable en este registro.</p>"; return; }
        const { T, n, nota } = plantilla(), DF = difs(caso, T);
        const orden = DF.map((d, l) => [l, d.score]).sort((a, b) => b[1] - a[1]);
        if (E.primera) { E.lead = orden[0][0]; E.primera = false; }   // al abrir, la derivación donde más difiere
        const top = orden[0][0], pico = DF[top].nd.indexOf(d3.max(DF[top].nd));
        const nOtra = D.registros.slice(1).filter((r) => r.label !== caso.label).length;
        const nMisma = D.registros.slice(1).filter((r) => r.beat && r.label === caso.label).length;
        const nTodos = D.registros.slice(1).filter((r) => r.beat).length;
        const pac = E.pac;
        $("#comp-titulo").textContent = `ECG ${caso.ecg_id} · real: ${NOM[caso.label] || caso.label}` +
            (pac ? ` · predicción de la red: ${NOM[pac.prediccion] || pac.prediccion}` : "");
        $("#comp-cuerpo").innerHTML = `
          <p class="c-resumen"><b>${nOtra} de 10</b> vecinos son de otra clase que el caso. La plantilla (línea azul) es la mediana del latido de ${nota} (${n}).
          La derivación en la que el caso más se separa de la plantilla es <b>${DERIV[top]}</b>, sobre todo en el tramo <b>${tramo(ms(pico))}</b> (aproximado).</p>
          <div class="c-ctrl">
            <label>Derivación <select id="c-lead">${DERIV.map((d, i) => `<option value="${i}" ${i === E.lead ? "selected" : ""}>${d}</option>`).join("")}</select></label>
            <label>Línea azul = mediana de <select id="c-ref">
              <option value="todos" ${E.ref === "todos" ? "selected" : ""}>los ${nTodos} vecinos</option>
              <option value="otra" ${E.ref === "otra" ? "selected" : ""} ${nOtra ? "" : "disabled"}>solo los vecinos de otra clase (${nOtra})</option>
              <option value="misma" ${E.ref === "misma" ? "selected" : ""} ${nMisma ? "" : "disabled"}>solo los vecinos de su misma clase (${nMisma})</option></select></label>
            <button id="c-auto" type="button" title="Dibuja en el gráfico grande, con línea punteada, todos los vecinos que forman la plantilla">Dibujar todos los de la plantilla</button>
            <button id="c-quitar" type="button" title="Quita del gráfico grande los vecinos dibujados">Quitar los dibujados</button>
          </div>
          <div class="c-ayuda">
            <p><b>¿Qué es la plantilla (línea azul)?</b> No es un ECG real. De cada vecino se toma su latido medio y la línea azul es, en cada instante, la <b>mediana</b> de esos latidos. Sirve para ver cómo es el latido «típico» de los vecinos y compararlo con el caso (rojo). La lista de arriba decide <b>qué vecinos entran</b> en esa mediana: todos, solo los de otra clase o solo los de su misma clase. Los que entran se ven resaltados abajo.</p>
            <p><b>Para ver un vecino concreto</b>, márcalo con «añadir al gráfico» (o usa el botón para dibujar todos los de la plantilla). Cada vecino tiene su propio color y trazo, y los que no entran en la plantilla no se pueden marcar.</p>
          </div>
          <div class="c-fila">
            <div class="c-grande"><div id="c-ley" class="c-ley"></div><div id="c-g"></div></div>
            <div class="c-barras"><p class="c-t">¿En qué derivaciones se separa más?</p><div id="c-b"></div></div>
          </div>
          <p class="c-t">Los 10 vecinos, derivación <span id="c-lname"></span> (el caso en rojo, de fondo)</p>
          <div id="c-grid" class="c-grid"></div>
          <details class="c-como"><summary>Cómo leer esta ventana</summary>
            <p>Cada ECG se resume en su <b>latido medio</b>: se alinean los latidos en el pico R (tiempo 0) y se toma la mediana, para que la comparación no dependa de en qué instante cae cada latido.
            Lo <b>naranja</b> marca los instantes en que el caso difiere de la plantilla en más del 20 % de la amplitud del latido de esa derivación.</p>
            <p>Esto muestra <b>dónde difieren las señales</b>; no prueba por qué la red se equivocó. Las barras resumen la diferencia media por derivación. El valor <i>r</i> de cada vecino es la correlación entre su latido medio y el del caso, con las 12 derivaciones juntas (1 = idéntico).</p></details>`;
        $("#c-lead").onchange = (e) => { E.lead = +e.target.value; dibujar(); };
        $("#c-ref").onchange = (e) => { E.ref = e.target.value; const { ks } = plantilla(); E.sel = E.sel.filter((k) => ks.has(k)); armar(); };
        $("#c-quitar").onclick = () => { E.sel = []; dibujar(); };
        $("#c-auto").onclick = () => {
            const { ks } = plantilla();
            E.sel = E.datos.registros.slice(1).filter((r) => r.beat && ks.has(r.k)).map((r) => r.k);
            dibujar();
        };
        dibujar();
    }

    function dibujar() {
        const D = E.datos, caso = D.registros[0], l = E.lead;
        const { T } = plantilla(), DF = difs(caso, T);
        $("#c-lname").textContent = DERIV[l];
        $("#c-lead").value = l;
        grande(D, caso, T, DF);
        barras(DF);
        rejilla(D, caso);
    }

    function ejes(svg, w, h, m, x, y, xl) {
        svg.append("g").attr("transform", `translate(0,${h - m.b})`).call(d3.axisBottom(x).ticks(8)).call((g) => g.selectAll("text").style("font-size", "10px"));
        svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5)).call((g) => g.selectAll("text").style("font-size", "10px"));
        if (xl) svg.append("text").attr("x", w - m.r).attr("y", h - 4).attr("text-anchor", "end").style("font-size", "10px").attr("fill", "#6b7280").text(xl);
    }

    function grande(D, caso, T, DF) {
        const el = $("#c-g"); el.innerHTML = "";
        const w = 640, h = 300, m = { l: 44, r: 10, t: 12, b: 28 }, l = E.lead;
        const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`).attr("width", "100%");
        const nums = [caso.beat[l], T[l], ...E.sel.map((k) => D.registros[k].beat[l])].flat();
        const x = d3.scaleLinear().domain([ms(0), ms(D.pre + D.post - 1)]).range([m.l, w - m.r]);
        const pad = (d3.max(nums) - d3.min(nums)) * 0.1 || 0.1;
        const y = d3.scaleLinear().domain([d3.min(nums) - pad, d3.max(nums) + pad]).range([h - m.b, m.t]);
        // zonas con diferencia
        const nd = DF[l].nd; let ini = null;
        const cierra = (a, b) => svg.append("rect").attr("x", x(ms(a))).attr("width", Math.max(2, x(ms(b)) - x(ms(a)))).attr("y", m.t).attr("height", h - m.t - m.b).attr("fill", "#f97316").attr("opacity", 0.22);
        nd.forEach((v, t) => { if (v > UMBRAL && ini === null) ini = t; if ((v <= UMBRAL || t === nd.length - 1) && ini !== null) { cierra(ini, t); ini = null; } });
        ejes(svg, w, h, m, x, y, "ms desde el pico R");
        svg.append("text").attr("x", 6).attr("y", m.t - 4).style("font-size", "10px").attr("fill", "#6b7280").text("mV");
        svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", m.t).attr("y2", h - m.b).attr("stroke", "#9ca3af").attr("stroke-dasharray", "3 3");
        const ln = d3.line().x((_, t) => x(ms(t))).y((v) => y(v));
        const trazo = (a, c, wd, dash) => svg.append("path").attr("d", ln(a)).attr("fill", "none").attr("stroke", c).attr("stroke-width", wd).attr("stroke-dasharray", dash || null);
        trazo(T[l], "#1e3a8a", 3.6);
        E.sel.forEach((k) => trazo(D.registros[k].beat[l], colNb(k), 2.1, dashNb(k)));
        trazo(caso.beat[l], "#d9372b", 2.5);
        // leyenda en HTML (admite muchos vecinos)
        const { n: nPl } = plantilla();
        const trozo = (c, t, dash, ancho) => `<span class="c-li"><svg width="26" height="8"><line x1="0" x2="26" y1="4" y2="4" stroke="${c}" stroke-width="${ancho || 2.4}" ${dash ? `stroke-dasharray="${dash}"` : ""}/></svg>${t}</span>`;
        $("#c-ley").innerHTML = trozo("#d9372b", "Caso") + trozo("#1e3a8a", `Plantilla (${nPl} ${nPl === 1 ? "vecino" : "vecinos"})`, null, 3.6) +
            E.sel.map((k) => trozo(colNb(k), `Vecino ${k}`, dashNb(k), 2.1)).join("") +
            `<span class="c-li"><i class="c-fr"></i>Difiere > 20 %</span>` +
            (nPl === 1 ? `<span class="c-li c-aviso1">Con un solo vecino, la plantilla coincide con él: dibújelo para verlo punteado sobre la línea azul.</span>` : "");
    }

    function barras(DF) {
        const el = $("#c-b"); el.innerHTML = "";
        const w = 250, h = 300, m = { l: 34, r: 30, t: 6, b: 6 };
        const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${w} ${h}`).attr("width", "100%");
        const mx = Math.max(0.1, d3.max(DF, (d) => d.score));
        const y = d3.scaleBand().domain(DERIV.map((_, i) => i)).range([m.t, h - m.b]).padding(0.2);
        const x = d3.scaleLinear().domain([0, mx]).range([0, w - m.l - m.r]);
        DERIV.forEach((d, i) => {
            const g = svg.append("g").style("cursor", "pointer").on("click", () => { E.lead = i; dibujar(); });
            g.append("rect").attr("x", 0).attr("y", y(i) - 1).attr("width", w).attr("height", y.bandwidth() + 2).attr("fill", i === E.lead ? "#e8eefb" : "transparent");
            g.append("text").attr("x", m.l - 6).attr("y", y(i) + y.bandwidth() / 2 + 4).attr("text-anchor", "end").style("font-size", "11px").style("font-weight", i === E.lead ? 700 : 400).text(d);
            g.append("rect").attr("x", m.l).attr("y", y(i)).attr("width", x(DF[i].score)).attr("height", y.bandwidth()).attr("fill", i === E.lead ? "#2b4a8b" : "#9fb1d6");
            g.append("text").attr("x", m.l + x(DF[i].score) + 4).attr("y", y(i) + y.bandwidth() / 2 + 4).style("font-size", "10px").attr("fill", "#6b7280").text(Math.round(DF[i].score * 100) + " %");
        });
    }

    function rejilla(D, caso) {
        const cont = $("#c-grid"); cont.innerHTML = "";
        const l = E.lead, todos = D.registros.filter((r) => r.beat).map((r) => r.beat[l]).flat();
        const { ks } = plantilla();
        const w = 220, h = 120, m = { l: 6, r: 6, t: 6, b: 6 };
        const x = d3.scaleLinear().domain([0, D.pre + D.post - 1]).range([m.l, w - m.r]);
        const y = d3.scaleLinear().domain([d3.min(todos), d3.max(todos)]).range([h - m.b, m.t]);
        const ln = d3.line().x((_, t) => x(t)).y((v) => y(v));
        D.registros.slice(1).forEach((r, i) => {
            const k = r.k, dist = E.dist[i], rr = r.beat ? corr(plano(caso.beat), plano(r.beat)) : null;
            const marcado = E.sel.indexOf(k);
            const card = document.createElement("div");
            const dentro = ks.has(k), filtra = E.ref !== "todos";
            card.className = "c-mini" + (marcado >= 0 ? " on" : "") + (filtra ? (dentro ? " dentro" : " fuera") : "");
            card.innerHTML = `<div class="c-cab" style="border-left:5px solid ${COL[r.label] || "#888"}"><b>Vecino ${k}</b> · ECG ${r.ecg_id}<br>
              ${NOM[r.label] || r.label}${dist !== undefined ? ` · dist ${(dist * 1000).toFixed(1)}×10⁻³` : ""}${rr !== null ? ` · r ${rr.toFixed(2)}` : ""}</div>
              ${filtra ? `<div class="c-tag ${dentro ? "si" : "no"}">${dentro ? "● en la plantilla" : "fuera de la plantilla"}</div>` : ""}
              <label class="c-chk"><input type="checkbox" ${marcado >= 0 ? "checked" : ""} ${!r.beat || (filtra && !dentro) ? "disabled" : ""}> añadir al gráfico
                <svg class="c-mu" width="30" height="8"><line x1="0" x2="30" y1="4" y2="4" stroke="${colNb(k)}" stroke-width="2.1" stroke-dasharray="${dashNb(k)}"/></svg></label>`;
            if (r.beat) {
                const svg = d3.select(card).insert("svg", "label").attr("viewBox", `0 0 ${w} ${h}`).attr("width", "100%");
                svg.append("line").attr("x1", x(D.pre)).attr("x2", x(D.pre)).attr("y1", m.t).attr("y2", h - m.b).attr("stroke", "#e5e7eb");
                svg.append("path").attr("d", ln(caso.beat[l])).attr("fill", "none").attr("stroke", "#d9372b").attr("stroke-width", 1.4).attr("opacity", .35);
                svg.append("path").attr("d", ln(r.beat[l])).attr("fill", "none").attr("stroke", COL[r.label] || "#555").attr("stroke-width", 1.8);
            } else card.insertAdjacentHTML("afterbegin", "<p class='c-nd'>Sin latido estable</p>");
            card.querySelector("input").onchange = (ev) => {
                if (ev.target.checked) { if (!E.sel.includes(k)) E.sel.push(k); } else E.sel = E.sel.filter((q) => q !== k);
                E.sel.sort((a, b) => a - b);
                dibujar();
            };
            cont.append(card);
        });
    }

    document.addEventListener("DOMContentLoaded", () => {
        $("#btn-comparar").addEventListener("click", abrir);
        $("#comp-cerrar").addEventListener("click", () => $("#dlg-comp").close());
        $("#btn-patron").addEventListener("click", () => $("#dlg-patron").showModal());
        $("#pat-cerrar").addEventListener("click", () => $("#dlg-patron").close());
    });
})();
