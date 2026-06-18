/**
 * ЧГКарта — city activity markers + player region scratch overlay.
 */
(function () {
  "use strict";

  const mapEl = document.getElementById("chgk-map");
  if (!mapEl) return;

  const boot = window.CHGK_MAP_BOOT || {};
  let map = null;
  let activeLayer = null;
  let inactiveLayer = null;
  let scratchLayer = null;
  const scratchStyle = {
    color: "#4338ca",
    weight: 1.2,
    fillColor: "#818cf8",
    fillOpacity: 0.5,
    opacity: 0.9,
  };
  const scratchHoverStyle = {
    weight: 2,
    fillOpacity: 0.72,
  };

  const playerSearch = document.getElementById("player-search");
  const playerIdInput = document.getElementById("player-id");
  const playerSearchClear = document.getElementById("player-search-clear");
  const playerSuggest = document.getElementById("player-suggest");
  let defaultVenueBounds = null;
  const scratchSummary = document.getElementById("scratch-summary");
  const scratchHoverTip = document.getElementById("scratch-hover-tip");
  const toggleActive = document.getElementById("toggle-active");
  const toggleInactive = document.getElementById("toggle-inactive");

  function debounce(fn, ms) {
    let t;
    return function (...args) {
      clearTimeout(t);
      t = setTimeout(() => fn.apply(this, args), ms);
    };
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function formatGameDate(iso) {
    if (!iso) return "";
    const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(iso));
    return m ? `${m[3]}.${m[2]}.${m[1]}` : String(iso);
  }

  function venuePopupHtml(c) {
    const place = [c.town_name, c.region_name, c.country_name]
      .filter(Boolean)
      .join(", ");
    const lines = [
      `<strong>${escapeHtml(c.town_name || "Город")}</strong>`,
      place !== c.town_name ? escapeHtml(place) : "",
      `Площадок: ${c.n_venues}`,
    ];
    if (c.last_game_date) {
      lines.push(`Последняя игра: ${formatGameDate(c.last_game_date)}`);
    }
    lines.push(`Ивентов: ${c.n_tournaments}`);
    lines.push(`Команд-игр: ${c.n_team_games}`);
    if (c.active) {
      lines.push(`За 60 дн.: ${c.n_recent} команд-игр`);
    }
    return lines.filter(Boolean).join("<br>");
  }

  function syncVenueLayers() {
    if (!map) return;
    const showActive = !toggleActive || toggleActive.checked;
    const showInactive = !toggleInactive || toggleInactive.checked;
    if (activeLayer) {
      if (showActive && !map.hasLayer(activeLayer)) map.addLayer(activeLayer);
      if (!showActive && map.hasLayer(activeLayer)) map.removeLayer(activeLayer);
    }
    if (inactiveLayer) {
      if (showInactive && !map.hasLayer(inactiveLayer)) map.addLayer(inactiveLayer);
      if (!showInactive && map.hasLayer(inactiveLayer)) map.removeLayer(inactiveLayer);
    }
  }

  function setVenueToggles(activeOn, inactiveOn) {
    if (toggleActive) toggleActive.checked = activeOn;
    if (toggleInactive) toggleInactive.checked = inactiveOn;
    syncVenueLayers();
  }

  function initMap() {
    map = L.map("chgk-map", { scrollWheelZoom: true }).setView([55.75, 37.62], 4);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>',
    }).addTo(map);

    map.createPane("scratch");
    map.getPane("scratch").style.zIndex = 350;
    map.createPane("venues");
    map.getPane("venues").style.zIndex = 450;

    scratchLayer = L.geoJSON(null, {
      pane: "scratch",
      interactive: true,
      style: () => ({ ...scratchStyle }),
    }).addTo(map);
    activeLayer = L.layerGroup([], { pane: "venues" });
    inactiveLayer = L.layerGroup([], { pane: "venues" });

    fetch("/api/map/venues")
      .then((r) => r.json())
      .then((cities) => {
        const bounds = [];
        cities.forEach((c) => {
          const isActive = c.active;
          const marker = L.circleMarker([c.lat, c.lon], {
            pane: "venues",
            radius: c.radius || (isActive ? 8 : 4),
            color: isActive ? "#047857" : "#64748b",
            weight: 1.5,
            fillColor: isActive ? "#10b981" : "#94a3b8",
            fillOpacity: isActive ? 0.75 : 0.4,
          });
          marker.bindPopup(venuePopupHtml(c));
          (isActive ? activeLayer : inactiveLayer).addLayer(marker);
          bounds.push([c.lat, c.lon]);
        });
        syncVenueLayers();
        if (bounds.length) {
          defaultVenueBounds = L.latLngBounds(bounds);
          map.fitBounds(defaultVenueBounds, { padding: [24, 24], maxZoom: 6 });
        }
        if (boot.playerScratch) {
          applyScratch(boot.playerScratch);
        }
      })
      .catch((err) => console.error("venues load failed", err));
  }

  const GEO_CACHE_BUST = "7";
  let boundariesPromise = null;
  function loadBoundaries() {
    if (!boundariesPromise) {
      const q = `?v=${GEO_CACHE_BUST}`;
      boundariesPromise = Promise.all([
        fetch(`/static/geo/ru_regions.geojson${q}`).then((r) => r.json()),
        fetch(`/static/geo/ua_regions.geojson${q}`).then((r) => r.json()),
        fetch(`/static/geo/countries.geojson${q}`).then((r) => r.json()),
      ]).then(([ru, ua, countries]) => ({ ru, ua, countries }));
    }
    return boundariesPromise;
  }

  function pointerXY(e) {
    const ev = (e && e.originalEvent) || e || {};
    return {
      x: typeof ev.clientX === "number" ? ev.clientX : 0,
      y: typeof ev.clientY === "number" ? ev.clientY : 0,
    };
  }

  function hideScratchTip() {
    if (!scratchHoverTip) return;
    scratchHoverTip.classList.add("scratch-tip-off");
    scratchHoverTip.setAttribute("aria-hidden", "true");
  }

  function showScratchTip(html, clientX, clientY) {
    if (!scratchHoverTip) return;
    scratchHoverTip.innerHTML = html;
    scratchHoverTip.classList.remove("scratch-tip-off");
    scratchHoverTip.setAttribute("aria-hidden", "false");
    scratchHoverTip.style.visibility = "hidden";
    scratchHoverTip.style.left = "0px";
    scratchHoverTip.style.top = "0px";
    const pad = 14;
    const rect = scratchHoverTip.getBoundingClientRect();
    let left = clientX + pad;
    let top = clientY + pad;
    if (left + rect.width > window.innerWidth - 8) {
      left = Math.max(8, clientX - rect.width - pad);
    }
    if (top + rect.height > window.innerHeight - 8) {
      top = Math.max(8, clientY - rect.height - pad);
    }
    scratchHoverTip.style.left = `${left}px`;
    scratchHoverTip.style.top = `${top}px`;
    scratchHoverTip.style.visibility = "visible";
  }

  function bindScratchLayerEvents(data, regionSet, countryIso, regionGames, countryGames) {
    scratchLayer.eachLayer((layer) => {
      const feature = layer.feature;
      if (!feature) return;
      const rid = feature.properties && feature.properties.rating_region_id;
      const iso = feature.properties && feature.properties.ISO_A2;
      let n = null;
      let title = "";
      if (rid != null && regionSet.has(Number(rid))) {
        n = regionGames[rid];
        title =
          (data.regions || []).find((r) => r.region_id === Number(rid))?.region_name || "";
      } else {
        const isoEh = feature.properties && feature.properties.ISO_A2_EH;
        const matchIso =
          (iso && countryIso.has(iso) && iso)
          || (isoEh && countryIso.has(isoEh) && isoEh);
        if (matchIso) {
          n = countryGames[matchIso];
          title =
            (data.countries || []).find((c) => c.iso_a2 === matchIso)?.country_name || "";
        }
      }
      if (n == null) return;
      const tipHtml = title
        ? `<strong>${escapeHtml(title)}</strong><br>${n} игр`
        : `${n} игр`;
      layer.on("mouseover", (e) => {
        layer.setStyle(scratchHoverStyle);
        const { x, y } = pointerXY(e);
        showScratchTip(tipHtml, x, y);
      });
      layer.on("mousemove", (e) => {
        const { x, y } = pointerXY(e);
        showScratchTip(tipHtml, x, y);
      });
      layer.on("mouseout", () => {
        layer.setStyle(scratchStyle);
        hideScratchTip();
      });
    });
  }

  function applyScratch(data) {
    if (!data || !scratchLayer) return;
    const regionSet = new Set((data.region_ids || []).map(Number));
    const countryIso = new Set(
      (data.countries || []).map((c) => c.iso_a2).filter(Boolean)
    );
    const regionGames = {};
    (data.regions || []).forEach((r) => {
      if (r.region_id > 0) regionGames[r.region_id] = r.n_games;
    });
    const countryGames = {};
    (data.countries || []).forEach((c) => {
      if (c.iso_a2) countryGames[c.iso_a2] = c.n_games;
    });

    loadBoundaries().then(({ ru, ua, countries }) => {
      const features = [];
      (ru.features || []).forEach((f) => {
        const rid = f.properties && f.properties.rating_region_id;
        if (rid != null && regionSet.has(Number(rid))) features.push(f);
      });
      (ua.features || []).forEach((f) => {
        const rid = f.properties && f.properties.rating_region_id;
        if (rid != null && regionSet.has(Number(rid))) features.push(f);
      });
      (countries.features || []).forEach((f) => {
        const props = f.properties || {};
        const iso = props.ISO_A2;
        const isoEh = props.ISO_A2_EH;
        if (
          (iso && countryIso.has(iso))
          || (isoEh && countryIso.has(isoEh))
        ) {
          features.push(f);
        }
      });

      scratchLayer.clearLayers();

      if (features.length) {
        // Leaflet GeoJSON.addData() ignores a second options arg — bind after add.
        scratchLayer.addData({ type: "FeatureCollection", features });
        bindScratchLayerEvents(data, regionSet, countryIso, regionGames, countryGames);
        setVenueToggles(false, false);
        try {
          map.fitBounds(scratchLayer.getBounds(), { padding: [20, 20], maxZoom: 6 });
        } catch (_) { /* empty */ }
      }
      if (scratchSummary) {
        const nReg = (data.regions || []).length;
        const nGames = data.scratch_games || 0;
        if (!nReg && !nGames) {
          scratchSummary.innerHTML =
            `<span class="font-medium">${escapeHtml(data.name)}</span> — нет sync-игр с площадкой`;
        } else {
          scratchSummary.innerHTML =
            `<span class="font-medium">${escapeHtml(data.name)}</span> — ` +
            `${nReg} регионов, ${nGames} игр на карте`;
        }
      }
    });
  }

  function syncPlayerClearBtn() {
    if (!playerSearchClear) return;
    const active = Boolean(
      (playerSearch && playerSearch.value.trim())
      || (playerIdInput && playerIdInput.value)
      || boot.playerScratch
    );
    playerSearchClear.classList.toggle("hidden", !active);
  }

  function clearScratch() {
    hideScratchTip();
    if (scratchLayer) scratchLayer.clearLayers();
    setVenueToggles(true, true);
    if (playerSearch) playerSearch.value = "";
    if (playerIdInput) playerIdInput.value = "";
    if (playerSuggest) playerSuggest.classList.add("hidden");
    boot.playerScratch = null;
    if (scratchSummary) {
      scratchSummary.textContent = "Выберите игрока, чтобы закрасить регионы на карте.";
    }
    if (map && defaultVenueBounds) {
      try {
        map.fitBounds(defaultVenueBounds, { padding: [24, 24], maxZoom: 6 });
      } catch (_) { /* empty */ }
    }
    const url = new URL(window.location.href);
    url.searchParams.delete("player_id");
    window.history.replaceState({}, "", url);
    syncPlayerClearBtn();
  }

  async function loadPlayerScratch(pid) {
    const res = await fetch(`/api/map/player/${pid}`);
    if (!res.ok) return;
    const data = await res.json();
    boot.playerScratch = data;
    applyScratch(data);
    syncPlayerClearBtn();
    const url = new URL(window.location.href);
    url.searchParams.set("player_id", String(pid));
    window.history.replaceState({}, "", url);
  }

  if (playerSearchClear) {
    playerSearchClear.addEventListener("click", () => clearScratch());
  }

  if (playerSearch) {
    playerSearch.addEventListener(
      "input",
      debounce(async () => {
        const q = playerSearch.value.trim();
        syncPlayerClearBtn();
        if (q.length < 2) {
          playerSuggest.classList.add("hidden");
          return;
        }
        const res = await fetch(`/api/map/players?q=${encodeURIComponent(q)}`);
        const rows = await res.json();
        if (!rows.length) {
          playerSuggest.classList.add("hidden");
          return;
        }
        playerSuggest.innerHTML = rows
          .map(
            (r) =>
              `<li class="px-3 py-2 hover:bg-indigo-50 cursor-pointer" data-pid="${r.player_id}">` +
              `${escapeHtml(r.name)} <span class="text-slate-400">#${r.player_id}</span></li>`
          )
          .join("");
        playerSuggest.classList.remove("hidden");
      }, 250)
    );

    playerSuggest.addEventListener("click", (e) => {
      const li = e.target.closest("[data-pid]");
      if (!li) return;
      const pid = Number(li.dataset.pid);
      playerIdInput.value = String(pid);
      playerSearch.value = li.textContent.replace(/#\d+.*/, "").trim();
      playerSuggest.classList.add("hidden");
      loadPlayerScratch(pid);
    });

    playerSearch.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        playerSuggest.classList.add("hidden");
        if (!playerSearch.value.trim()) clearScratch();
      }
    });
  }

  if (toggleActive) toggleActive.addEventListener("change", syncVenueLayers);
  if (toggleInactive) toggleInactive.addEventListener("change", syncVenueLayers);

  syncPlayerClearBtn();
  initMap();
})();
