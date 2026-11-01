const form = document.querySelector('#prediction-form');
const home = document.querySelector('#home-team');
const away = document.querySelector('#away-team');
const errorBox = document.querySelector('#error');
const result = document.querySelector('#result');
const submitButton = form.querySelector('button[type="submit"]');

function option(team) { const el = document.createElement('option'); el.value = team; el.textContent = team; return el; }
async function loadTeams() {
  const response = await fetch('/api/teams');
  const data = await response.json();
  data.teams.forEach(team => { home.append(option(team)); away.append(option(team)); });
  home.value = data.teams.includes('Turkey') ? 'Turkey' : data.teams[0];
  away.value = data.teams.includes('Germany') ? 'Germany' : data.teams[1];
}
function pct(value) { return `${(value * 100).toFixed(2)}%`; }
function probabilityRow(label, value) {
  return `<div class="probability"><span>${label}</span><div class="bar"><i style="width:${value * 100}%"></i></div><strong>${pct(value)}</strong></div>`;
}
function render(data) {
  document.querySelector('#result-home').textContent = home.value;
  document.querySelector('#result-away').textContent = away.value;
  document.querySelector('#xg-home').textContent = data.expected_home_goals.toFixed(2);
  document.querySelector('#xg-away').textContent = data.expected_away_goals.toFixed(2);
  document.querySelector('#probabilities').innerHTML = [
    probabilityRow(`${home.value} kazanır`, data.home_win_probability),
    probabilityRow('Beraberlik', data.draw_probability),
    probabilityRow(`${away.value} kazanır`, data.away_win_probability)
  ].join('');
  document.querySelector('#scores').innerHTML = data.top_score_probabilities.map(item => `<div class="score"><strong>${item.score}</strong><span>${pct(item.probability)}</span></div>`).join('');
  document.querySelector('#meta').textContent = `Model: ${data.model} · Veri tarihi: ${data.as_of_date} · FIFA snapshot: ${data.fifa_snapshot_home} / ${data.fifa_snapshot_away}`;
  result.classList.remove('hidden');
  result.scrollIntoView({behavior:'smooth', block:'start'});
}
document.querySelector('#swap').addEventListener('click', () => { [home.value, away.value] = [away.value, home.value]; });
form.addEventListener('submit', async event => {
  event.preventDefault(); errorBox.classList.add('hidden'); submitButton.disabled = true; submitButton.textContent = 'Hesaplanıyor…';
  try {
    const response = await fetch('/api/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({ home_team:home.value, away_team:away.value, tournament:document.querySelector('#tournament').value, neutral:document.querySelector('#neutral').checked }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Tahmin oluşturulamadı.');
    render(data);
  } catch (error) { errorBox.textContent = error.message; errorBox.classList.remove('hidden'); result.classList.add('hidden'); }
  finally { submitButton.disabled = false; submitButton.textContent = 'Tahmini Hesapla'; }
});
loadTeams().catch(error => { errorBox.textContent = `Takımlar yüklenemedi: ${error.message}`; errorBox.classList.remove('hidden'); });
