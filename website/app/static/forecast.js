/**
 * Forecast UI: tournament pack picker, player chips + autocomplete,
 * client-side filter on the upcoming-tournaments table.
 */
(function () {
  "use strict";

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

  function hideList(ul) {
    ul.classList.add("hidden");
    ul.innerHTML = "";
  }

  function showList(ul) {
    ul.classList.remove("hidden");
  }

  // -------------------------------------------------------------------------
  // Tournament autocomplete (past packs + «прогноз на прошедшем»)
  // -------------------------------------------------------------------------

  function initTournamentPicker(root) {
    const search = root.querySelector("[data-tournament-search]");
    const hidden = root.querySelector("[data-tournament-id]");
    const results = root.querySelector("[data-tournament-results]");
    const hint = root.querySelector("[data-tournament-selected]");
    if (!search || !results) return;

    const mode = root.dataset.tournamentPicker || "pack"; // pack | link

    const fetchMatches = debounce(async () => {
      const q = search.value.trim();
      if (q.length < 2) {
        hideList(results);
        return;
      }
      try {
        const res = await fetch(
          "/api/forecast/tournaments?q=" + encodeURIComponent(q)
        );
        const items = await res.json();
        results.innerHTML = "";
        if (!items.length) {
          results.innerHTML =
            '<li class="px-3 py-2 text-sm text-slate-500">Ничего не найдено</li>';
          showList(results);
          return;
        }
        items.forEach((t) => {
          const li = document.createElement("li");
          li.className =
            "px-3 py-2 text-sm cursor-pointer hover:bg-indigo-50 border-b border-slate-100 last:border-0";
          const label = escapeHtml(t.title) + " · " + escapeHtml(t.date || "—");
          li.innerHTML =
            '<span class="font-medium text-slate-900">' +
            label +
            '</span> <span class="text-slate-500 num">#' +
            t.tournament_id +
            "</span>";
          li.addEventListener("mousedown", (e) => {
            e.preventDefault();
            if (mode === "link") {
              window.location.href =
                "/forecast/tournament/" + t.tournament_id;
              return;
            }
            if (hidden) hidden.value = String(t.tournament_id);
            search.value = t.title;
            if (hint) {
              hint.textContent =
                "Выбран: " + t.title + " (#" + t.tournament_id + ")";
              hint.classList.remove("hidden");
            }
            const packKind = root.closest("form")?.querySelector(
              'select[name="pack_kind"]'
            );
            if (packKind) packKind.value = "past";
            hideList(results);
          });
          results.appendChild(li);
        });
        showList(results);
      } catch (_) {
        hideList(results);
      }
    }, 250);

    search.addEventListener("input", fetchMatches);
    search.addEventListener("focus", fetchMatches);
    search.addEventListener("blur", () => {
      setTimeout(() => hideList(results), 150);
    });
  }

  // -------------------------------------------------------------------------
  // Player autocomplete (team builder)
  // -------------------------------------------------------------------------

  function initPlayerPicker(root) {
    const search = root.querySelector("[data-player-search]");
    const hidden = root.querySelector("[data-player-ids]");
    const chips = root.querySelector("[data-player-chips]");
    const results = root.querySelector("[data-player-results]");
    if (!search || !hidden || !chips || !results) return;

    let ids = [];
    const initial = (hidden.value || "").trim();
    if (initial) {
      ids = initial
        .split(/[,;\s]+/)
        .map((x) => parseInt(x, 10))
        .filter((n) => n > 0);
    }

    const chipData = new Map();

    function syncHidden() {
      hidden.value = ids.join(",");
    }

    function renderChips() {
      chips.innerHTML = "";
      ids.forEach((pid) => {
        const meta = chipData.get(pid) || { name: "#" + pid, theta: null };
        const chip = document.createElement("span");
        chip.className =
          "inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-indigo-100 text-indigo-900 text-sm";
        const theta =
          meta.theta != null
            ? " θ=" + Number(meta.theta).toFixed(2)
            : "";
        chip.innerHTML =
          escapeHtml(meta.name) +
          '<span class="text-indigo-600/80 num text-xs">' +
          theta +
          "</span>" +
          '<button type="button" class="ml-0.5 text-indigo-700 hover:text-rose-600" aria-label="Удалить">×</button>';
        chip.querySelector("button").addEventListener("click", () => {
          ids = ids.filter((x) => x !== pid);
          syncHidden();
          renderChips();
        });
        chips.appendChild(chip);
      });
    }

    async function hydrateFromServer() {
      if (!ids.length) return;
      try {
        const res = await fetch(
          "/api/forecast/players?ids=" + ids.join(",")
        );
        const rows = await res.json();
        rows.forEach((p) => {
          chipData.set(p.player_id, p);
        });
        renderChips();
      } catch (_) {
        renderChips();
      }
    }

    function addPlayer(p) {
      if (ids.includes(p.player_id)) return;
      if (ids.length >= 12) return;
      ids.push(p.player_id);
      chipData.set(p.player_id, p);
      syncHidden();
      renderChips();
      search.value = "";
      hideList(results);
    }

    const fetchMatches = debounce(async () => {
      const q = search.value.trim();
      if (q.length < 2) {
        hideList(results);
        return;
      }
      try {
        const res = await fetch(
          "/api/forecast/players?q=" + encodeURIComponent(q)
        );
        const items = await res.json();
        results.innerHTML = "";
        if (!items.length) {
          results.innerHTML =
            '<li class="px-3 py-2 text-sm text-slate-500">Никого не найдено</li>';
          showList(results);
          return;
        }
        items.forEach((p) => {
          const li = document.createElement("li");
          li.className =
            "px-3 py-2 text-sm cursor-pointer hover:bg-indigo-50 border-b border-slate-100 last:border-0";
          const name = escapeHtml(p.name);
          const th =
            p.theta != null
              ? " <span class='num text-slate-500'>" +
                Number(p.theta).toFixed(2) +
                "</span>"
              : "";
          li.innerHTML =
            name +
            th +
            " <span class='text-slate-400 num'>#" +
            p.player_id +
            "</span>";
          li.addEventListener("mousedown", (e) => {
            e.preventDefault();
            addPlayer(p);
          });
          results.appendChild(li);
        });
        showList(results);
      } catch (_) {
        hideList(results);
      }
    }, 200);

    search.addEventListener("input", fetchMatches);
    search.addEventListener("keydown", (e) => {
      if (e.key === "Enter") e.preventDefault();
    });
    search.addEventListener("blur", () => {
      setTimeout(() => hideList(results), 150);
    });

    hydrateFromServer();
  }

  // -------------------------------------------------------------------------
  // Upcoming table filter
  // -------------------------------------------------------------------------

  function initUpcomingFilter() {
    const input = document.querySelector("[data-upcoming-filter]");
    const tbody = document.querySelector("[data-upcoming-tbody]");
    if (!input || !tbody) return;
    const rows = Array.from(tbody.querySelectorAll("tr"));
    input.addEventListener("input", () => {
      const q = input.value.trim().toLowerCase();
      rows.forEach((tr) => {
        const text = (tr.dataset.searchText || tr.textContent || "").toLowerCase();
        tr.classList.toggle("hidden", q.length > 0 && !text.includes(q));
      });
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    document
      .querySelectorAll("[data-tournament-picker]")
      .forEach(initTournamentPicker);
    document.querySelectorAll("[data-player-picker]").forEach(initPlayerPicker);
    initUpcomingFilter();
  });
})();
