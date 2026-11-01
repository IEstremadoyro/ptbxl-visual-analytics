/* Ventana «Pruebas de apoyo»: resultados de análisis calculados fuera de la plataforma (notebook)
   sobre los 1.650 registros de prueba. Aquí solo se muestran; no se recalculan al filtrar. */
(() => {
    "use strict";
    const $ = (s) => document.querySelector(s);
    const f3 = (x) => x.toFixed(3).replace(".", ",");

    const DIST = [  // [método, exactitud, infartos detectados (de 256), vecinos en común aprox. de 10]
        ["Red neuronal (la que clasifica en la plataforma)", 0.758, 173, null],
        ["Vecinos, 64 números, coseno (el de la plataforma)", 0.776, 153, 10],
        ["Vecinos, 64 números, Manhattan", 0.778, 156, 9],
        ["Vecinos, 64 números, Chebyshev", 0.772, 146, 7],
        ["Vecinos, 128 números, coseno", 0.785, 158, 5],
        ["Vecinos, 128 números, euclidiana", 0.778, 145, 4],
        ["Vecinos, 128 números, Manhattan", 0.783, 144, 4],
    ];
    const KS = [[1, 142], [3, 147], [5, 150], [10, 153], [20, 149], [50, 149]];
    const PRE = [  // [variante, infartos detectados, perdidos que se recuperan, detectados que se pierden]
        ["Sin cambio", 173, 0, 0],
        ["Filtro pasa-bajas 25 Hz", 176, 8, 5],
        ["Filtro pasa-altas 1 Hz", 175, 6, 4],
        ["Ruido gaussiano", 174, 3, 2],
        ["Amplitud × 0,8", 171, 7, 9],
        ["Amplitud × 1,2", 172, 2, 3],
    ];

    function barra(v, max, col) {
        return `<div class="pb-barra"><span style="width:${Math.round((v / max) * 100)}%;background:${col}"></span></div>`;
    }

    const PEST = {
        dist: () => `
          <p class="pb-idea"><b>En una frase:</b> cambiar la forma de medir qué tan cerca está un ECG de otro casi no cambia cuántos infartos se detectan, y ninguna variante supera a la red.</p>
          <p class="pb-p">Probamos buscar los 10 vecinos con otras «reglas de distancia» y con más información (128 números en vez de 64). Cada fila dice cuántos de los 256 infartos de prueba se detectan.</p>
          <table class="pb-t"><thead><tr><th>Método</th><th>Exactitud general</th><th colspan="2">Infartos detectados (de 256)</th><th>Vecinos en común con el método de la plataforma (aprox., de 10)</th></tr></thead><tbody>
          ${DIST.map((r, i) => `<tr class="${i === 0 ? "pb-ref" : ""}"><td>${r[0]}</td><td>${f3(r[1])}</td><td><b>${r[2]}</b></td><td class="pb-b">${barra(r[2], 256, i === 0 ? "#2b4a8b" : "#9fb1d6")}</td><td>${r[3] === null ? "–" : r[3]}</td></tr>`).join("")}
          </tbody></table>
          <p class="pb-n">Coseno y euclidiana dan los mismos vecinos porque el resumen de 64 números está normalizado. El número de vecinos en común se estimó a partir del índice de Jaccard.</p>
          <p class="pb-p"><b>¿Y si se vota con más o menos vecinos (K)?</b> Con 10 vecinos sale lo mejor, pero la diferencia es pequeña:</p>
          <div class="pb-ks">${KS.map(([k, v]) => `<div><div class="pb-kbar" style="height:${Math.round(v * 0.6)}px"></div><b>${v}</b><br>K = ${k}</div>`).join("")}</div>
          <p class="pb-n">Infartos detectados (de 256) según el número de vecinos que votan. Las distancias alternativas de cada caso se ven en la ficha del paciente.</p>`,

        lej: () => `
          <p class="pb-idea"><b>En una frase:</b> los infartos que la red pierde están rodeados de vecinos más lejanos y casi todos de otra clase.</p>
          <div class="pb-dos">
            <div><h4>¿Qué tan lejos están sus vecinos?</h4>
              <div class="pb-fila"><span>Infartos detectados (173)</span>${barra(2.83, 4, "#3fa66b")}<b>2,83</b></div>
              <div class="pb-fila"><span>Infartos perdidos (83)</span>${barra(3.69, 4, "#d9372b")}<b>3,69</b></div>
              <div class="pb-fila"><span>Otras clases (1.394)</span>${barra(3.03, 4, "#9ca3af")}<b>3,03</b></div>
              <p class="pb-n">Distancia media a los 10 vecinos. Los perdidos tienen vecinos un 30 % más lejos. Los números solo sirven para comparar entre sí.</p></div>
            <div><h4>De sus 10 vecinos, ¿cuántos también son infarto?</h4>
              <div class="pb-fila"><span>Infartos detectados</span>${barra(7.6, 10, "#3fa66b")}<b>7,6</b></div>
              <div class="pb-fila"><span>Infartos perdidos</span>${barra(1.3, 10, "#d9372b")}<b>1,3</b></div>
              <p class="pb-n">Promedio de vecinos (de 10) que son infarto. En los perdidos, los vecinos votan NORM en 48 de los 83 casos y STTC en 25: el ECG cae en una zona que la red asocia con otra clase.</p></div>
          </div>`,

        agr: () => `
          <p class="pb-idea"><b>En una frase:</b> si se agrupan los ECG sin usar las etiquetas, casi todos los infartos perdidos caen en grupos de otra clase.</p>
          <div class="pb-dos">
            <div><h4>K-means (5 grupos)</h4>
              <p class="pb-p">Un grupo reúne a casi todos los infartos detectados (172 de 173). Los 83 perdidos quedan repartidos así, según el grupo que los contiene:</p>
              <div class="pb-pila"><span style="flex:44;background:#3fa66b">NORM 44</span><span style="flex:30;background:#f29b38">STTC 30</span><span style="flex:5;background:#6fc3e3">CD 5</span><span style="flex:4;background:#d9372b">MI 4</span></div>
              <p class="pb-n">79 de 83 caen en grupos de otra clase (NORM, STTC o CD); solo 4 quedan en un grupo de infarto.</p></div>
            <div><h4>DBSCAN</h4>
              <p class="pb-p">No aportó: casi todos los registros (80 %) cayeron en un solo grupo, así que no separa nada.</p>
              <p class="pb-n">Solo se probaron tres ajustes; eso indica que los infartos no forman un grupo aparte, pero no descarta otros ajustes.</p></div>
          </div>
          <p><button id="pb-mapa" type="button" class="pb-btn">Ver en el mapa: color por grupo de K-means y solo los infartos mal clasificados</button></p>`,

        pre: () => `
          <p class="pb-idea"><b>En una frase:</b> filtrar la señal, añadirle ruido o cambiar su amplitud casi no cambia el resultado; los errores no desaparecen.</p>
          <p class="pb-p">Se modificó la señal que entra a la red ya entrenada (sin reentrenarla) y se contó cuántos de los 256 infartos se detectan.</p>
          <table class="pb-t"><thead><tr><th>Cambio en la señal</th><th colspan="2">Infartos detectados (de 256)</th><th>Perdidos que ahora se detectan (de 83)</th><th>Detectados que ahora se pierden (de 173)</th></tr></thead><tbody>
          ${PRE.map((r, i) => `<tr class="${i === 0 ? "pb-ref" : ""}"><td>${r[0]}</td><td><b>${r[1]}</b></td><td class="pb-b">${barra(r[1], 256, i === 0 ? "#2b4a8b" : "#9fb1d6")}</td><td>${r[2]}</td><td>${r[3]}</td></tr>`).join("")}
          </tbody></table>
          <p class="pb-n">Esto mide cuánto depende el resultado de la entrada; no dice qué pasaría si la red se entrenara con otro preprocesamiento.</p>`,
    };
    const TABS = [["dist", "Distancias"], ["lej", "Vecinos lejanos"], ["agr", "Agrupamientos"], ["pre", "Cambios en la señal"]];

    function mostrar(k) {
        $("#pru-cuerpo").innerHTML = `
          <p class="pat-aviso"><b>Cómo leer esta ventana.</b> Son pruebas hechas aparte, en el notebook, con los 1.650 registros de prueba. Aquí solo se muestran. Describen lo observado en un solo entrenamiento; no prueban las causas.</p>
          <div class="pb-tabs">${TABS.map(([id, t]) => `<button type="button" data-t="${id}" class="${id === k ? "on" : ""}">${t}</button>`).join("")}</div>
          <div class="pb-pane">${PEST[k]()}</div>`;
        document.querySelectorAll(".pb-tabs button").forEach((b) => b.onclick = () => mostrar(b.dataset.t));
        const bm = $("#pb-mapa");
        if (bm) bm.onclick = () => {
            $("#dlg-pruebas").close();
            $("#btn-limpiar").click();   // parte de un mapa sin selecciones
            const sel = $("#s-color"); sel.value = "km"; sel.dispatchEvent(new Event("change", { bubbles: true }));
            const fr = $("#f-front"); if (!fr.checked) fr.click();
        };
    }

    document.addEventListener("DOMContentLoaded", () => {
        $("#btn-pruebas").addEventListener("click", () => { mostrar("dist"); $("#dlg-pruebas").showModal(); });
        $("#pru-cerrar").addEventListener("click", () => $("#dlg-pruebas").close());
    });
})();
