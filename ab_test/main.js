document.addEventListener('DOMContentLoaded', async () => {
  const head = document.getElementById('site-header');
  const foot = document.getElementById('site-footer');
  if (head) head.innerHTML = await fetch('/partials/header.html').then(r => r.text());
  if (foot) foot.innerHTML = await fetch('/partials/footer.html').then(r => r.text());

  const path = location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('nav a').forEach(a => {
    if (a.getAttribute('href') === path) a.setAttribute('aria-current', 'page');
  });

  if (document.title.startsWith('Projects') || document.getElementById('projects-app')) {
    bootProjectsApp();
  }
});

function bootProjectsApp() {

const BANDS = [
  { max: 0.30, text: 'UNLIKELY CLICKBAIT',      cls: 'risk-low' },
  { max: 0.60, text: 'LIKELY CLICKBAIT',        cls: 'risk-med' },
  { max: 0.75, text: 'HIGHLY LIKELY CLICKBAIT', cls: 'risk-high' },
  { max: 1.01, text: 'CLICKBAIT',               cls: 'risk-cb' },
];
const riskBand = p => BANDS.find(b => p < b.max);


  const titleEl    = document.getElementById('acbd-title') || document.querySelector('#projects-app .titlebar h1');
  const startLink  = document.getElementById('start-link');
  const assignNote = document.getElementById('assign-note');
  const trialForm  = document.getElementById('trial');
  const listEl     = document.getElementById('article-list');
  const roundTitle = document.getElementById('round-title');
  const nextLink   = document.getElementById('next-link');
  const submitLink = document.getElementById('submit-link');

  const ARTICLES = [
    {
      "title": "Amy Schumer Buys Back Her Dad's Farm ",
      "isClickbait": 0,
      "likelihood": 0.03,
      "id": "a01"
    },
    {
      "title": "How The Shale Boom Turned The World Upside Down",
      "isClickbait": 1,
      "likelihood": 0.18,
      "id": "a02"
    },
    {
      "title": "Democrats Elect Tom Perez as Party Chairman",
      "isClickbait": 0,
      "likelihood": 0.07,
      "id": "a03"
    },
    {
      "title": "Win one of 12 brilliant designer gifts worth \u00a37,150 - and help the Telegraph's charity appeal\u00a0",
      "isClickbait": 1,
      "likelihood": 0.65,
      "id": "a04"
    },
    {
      "title": "Feds Abandon Effort To Force Twitter To Identify Owner Of Anonymous Anti-Trump Account",
      "isClickbait": 0,
      "likelihood": 0.04,
      "id": "a05"
    },
    {
      "title": "Here's how LaCroix sparkling water gained a cult following",
      "isClickbait": 1,
      "likelihood": 0.42,
      "id": "a06"
    },
    {
      "title": "What To Say To The Millennials Who Want To Save The World",
      "isClickbait": 1,
      "likelihood": 0.75,
      "id": "a07"
    },
    {
      "title": "The clever pun pictures (that aren't as easy as you think) baffling the Internet... so how many can YOU solve?",
      "isClickbait": 1,
      "likelihood": 0.95,
      "id": "a08"
    },
    {
      "title": "Inside Uber\u2019s Aggressive, Unrestrained Workplace Culture ",
      "isClickbait": 0,
      "likelihood": 0.04,
      "id": "a09"
    },
    {
      "title": "Inside the world's only 'flying eye hospital'",
      "isClickbait": 1,
      "likelihood": 0.56,
      "id": "a10"
    },
    {
      "title": "16 Gifts For People Who Always Need To Charge Their\u00a0Phone",
      "isClickbait": 1,
      "likelihood": 0.81,
      "id": "a11"
    },
    {
      "title": "Mourners gather for funeral of black victim of sword slaying ",
      "isClickbait": 1,
      "likelihood": 0.01,
      "id": "a12"
    },
    {
      "title": "A Look at Trump's Cabinet Picks ",
      "isClickbait": 0,
      "likelihood": 0.07,
      "id": "a13"
    },
    {
      "title": "100 Pictures That\u2019ll Make You Pee Yourself\u00a0Laughing",
      "isClickbait": 1,
      "likelihood": 0.97,
      "id": "a14"
    },
    {
      "title": "At least 30 killed in attack on Kabul military hospital",
      "isClickbait": 0,
      "likelihood": 0.0,
      "id": "a15"
    },
    {
      "title": "Inauguration Day weather forecast: gray and wet ",
      "isClickbait": 0,
      "likelihood": 0.02,
      "id": "a16"
    },
    {
      "title": "Here\u2019s Why It\u2019s Legal for Airlines to Kick You Off Your Flight",
      "isClickbait": 1,
      "likelihood": 0.64,
      "id": "a17"
    },
    {
      "title": "WATCH: Heart-Stopping Rescue As Kitten Comes Within A Whisker Of Death",
      "isClickbait": 1,
      "likelihood": 0.74,
      "id": "a18"
    },
    {
      "title": "Pence tries to reassure European leaders shaken by Trump",
      "isClickbait": 0,
      "likelihood": 0.01,
      "id": "a19"
    },
    {
      "title": "Day In The Life: Balancing Productivity And Creativity",
      "isClickbait": 1,
      "likelihood": 0.61,
      "id": "a20"
    },
    {
      "title": "Rex Tillerson only took Secretary of State job 'because his wife said God wanted him to'",
      "isClickbait": 0,
      "likelihood": 0.02,
      "id": "a21"
    },
    {
      "title": "Japan slaughters 333 whales for meat that will end up in school dinners in annual Antarctic hunt which flouts worldwide ban\u00a0",
      "isClickbait": 0,
      "likelihood": 0.01,
      "id": "a22"
    },
    {
      "title": "Istanbul stadium attack: BBC witnesses anger at scene",
      "isClickbait": 0,
      "likelihood": 0.01,
      "id": "a23"
    },
    {
      "title": "Lady Gaga's Sales Surge 1,000 Percent on Heels of Super Bowl Performance ",
      "isClickbait": 0,
      "likelihood": 0.02,
      "id": "a24"
    },
    {
      "title": "Early Trends Put BJP In Firm Control Of UP, Uttarakhand And Manipur, Congress Leads In Punjab, Goa",
      "isClickbait": 0,
      "likelihood": 0.11,
      "id": "a25"
    },
    {
      "title": "The Madness Is Here",
      "isClickbait": 0,
      "likelihood": 0.29,
      "id": "a26"
    },
    {
      "title": "How a white kid\u2019s taunt and a black student\u2019s body slam made race their high school\u2019s main subject",
      "isClickbait": 1,
      "likelihood": 0.51,
      "id": "a27"
    },
    {
      "title": "Turkish president's supporters burn a FRENCH flag after mistaking it for Holland's and Norwegian woman mistaken for Dutch is threatened as diplomatic row between countries escalates\u00a0",
      "isClickbait": 0,
      "likelihood": 0.01,
      "id": "a28"
    },
    {
      "title": "Six Records That Show There Will Be Only One Don Bradman",
      "isClickbait": 1,
      "likelihood": 0.43,
      "id": "a29"
    },
    {
      "title": "Rajat Kapoor Feels Movies Made In 1950s Were The Best That Indian Cinema Has Ever Seen!",
      "isClickbait": 1,
      "likelihood": 0.05,
      "id": "a30"
    }
  ];

  const showStart = () => startLink && startLink.classList.remove('hidden');
  if (titleEl) {
    titleEl.addEventListener('click', showStart);
    titleEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); showStart(); }
    });
  }

  let uid = null;
  let group = null;
  let rounds = []; 
  let currentRound = 0;
  const selectedByRound = [new Set(), new Set(), new Set()];
  const selectedAll = new Set();

  function activate(el, fn) {
    if (!el) return;
    const run = (e) => {
      if (el.hidden) return;
      if (el.getAttribute('aria-disabled') === 'true') return;
      fn(e);
    };
    el.addEventListener('click', run);
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); run(e); }
    });
  }

  if (startLink) {
    startLink.addEventListener('click', () => {
      if (startLink.getAttribute('aria-disabled') === 'true') return;

      uid = nextUID();
      group = assignGroupAlternating();

      assignNote.textContent = `Your ID: ${uid}`;

      rounds = splitIntoRounds(ARTICLES);
      currentRound = 0;
      selectedByRound.forEach(s => s.clear());
      selectedAll.clear();

      startLink.setAttribute('aria-disabled', 'true');
      if (trialForm) {
        trialForm.hidden = false;
        renderRound();
      }
    });
  }

  if (listEl) {
    listEl.addEventListener('change', (e) => {
      const box = e.target;
      if (!box || box.type !== 'checkbox') return;
      const id = box.value;
      const s = selectedByRound[currentRound];

      if (box.checked) {
        if (s.size >= 5) { box.checked = false; return; }
        s.add(id);
        selectedAll.add(id);
      } else {
        s.delete(id);
        selectedAll.delete(id);
      }
      enforceRoundLimit();
      updateButtons();
    });
  }

  activate(nextLink, () => {
    if (selectedByRound[currentRound].size !== 5) return;
    if (currentRound < 2) {
      currentRound += 1;
      renderRound();
    }
  });

  activate(submitLink, finalizeAndSave);

  function finalizeAndSave() {
    if (![0,1,2].every(r => selectedByRound[r].size === 5)) return;

    const flat = rounds.flat();
    const map = Object.fromEntries(flat.map(a => [a.id, a]));
    const chosen = [...selectedAll].map(id => map[id]);
    const cb = chosen.filter(a => a.isClickbait === 1).length;
    const ncb = chosen.length - cb;

    const rec = {
      uid,
      group,
      chosenIds: [...selectedAll],
      counts: { clickbait: cb, nonClickbait: ncb },
      at: new Date().toISOString()
    };
    const all = JSON.parse(localStorage.getItem('acbd_records') || '[]');
    all.push(rec);
    localStorage.setItem('acbd_records', JSON.stringify(all));

    assignNote.textContent = 'THANK U';
    if (trialForm) trialForm.hidden = true;
    if (startLink) startLink.removeAttribute('aria-disabled');
  }

  function renderRound() {
    const items = rounds[currentRound];
    if (roundTitle) roundTitle.textContent = `ROUND ${currentRound + 1}`;
    if (listEl) listEl.innerHTML = '';
  
    items.forEach(a => {
      const checked = selectedByRound[currentRound].has(a.id) ? 'checked' : '';
      let labelExtra = '';
      if (group === 'experimental') {
        const band = riskBand(a.likelihood);
        const pct  = Math.round(a.likelihood * 100);
        labelExtra = `<span class="badge ${band.cls}">${band.text} • ${pct}%</span>`;
      }
      const li = document.createElement('li');
      li.innerHTML = `
        <label>
          <input type="checkbox" value="${a.id}" ${checked} />
          <span>${escapeHTML(a.title)}</span>
          ${labelExtra}
        </label>
      `;
      listEl.appendChild(li);
    });
  
    enforceRoundLimit();
    updateButtons();
  }  

  function updateButtons() {
    if (!nextLink || !submitLink) return;
    const sSize = selectedByRound[currentRound].size;

    if (currentRound < 2) {
      nextLink.hidden = false;
      submitLink.hidden = true;
      nextLink.setAttribute('aria-disabled', sSize === 5 ? 'false' : 'true');
    } else {
      nextLink.hidden = true;
      submitLink.hidden = false;
      const ok = selectedByRound[0].size === 5 && selectedByRound[1].size === 5 && selectedByRound[2].size === 5;
      submitLink.setAttribute('aria-disabled', ok ? 'false' : 'true');
    }
  }

  function enforceRoundLimit() {
    if (!listEl) return;
    const s = selectedByRound[currentRound];
    const atLimit = s.size >= 5;
    const boxes = listEl.querySelectorAll('input[type="checkbox"]');
    boxes.forEach(b => {
      if (s.has(b.value)) { b.disabled = false; return; }
      b.disabled = atLimit;
    });
  }

  function splitIntoRounds(pool) {
    const cb  = shuffle(pool.filter(a => a.isClickbait === 1));
    const ncb = shuffle(pool.filter(a => a.isClickbait === 0));

    const r1 = shuffle(cb.slice(0,5).concat(ncb.slice(0,5)));
    const r2 = shuffle(cb.slice(5,10).concat(ncb.slice(5,10)));
    const r3 = shuffle(cb.slice(10,15).concat(ncb.slice(10,15)));
    return [r1, r2, r3];
  }

  function nextUID() {
    const n = Number(localStorage.getItem('acbd_uid_counter') || '0') + 1;
    localStorage.setItem('acbd_uid_counter', String(n));
    return String(n).padStart(3, '0');
  }

  function assignGroupAlternating() {
    const last = localStorage.getItem('acbd_last_group') || 'experimental';
    const next = last === 'experimental' ? 'control' : 'experimental';
    localStorage.setItem('acbd_last_group', next);
    return next;
  }

  function shuffle(a0) {
    const a = a0.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  function escapeHTML(s) {
    return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }
}
