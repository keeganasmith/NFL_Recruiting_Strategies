const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

const sharedPath = path.join(__dirname, '..', 'shared.js');
const sharedCode = fs.readFileSync(sharedPath, 'utf8');
const context = { console };
vm.createContext(context);
vm.runInContext(sharedCode, context, { filename: 'shared.js' });

const {
  selectSportsRefSlugSource,
  normalizeSlugBase,
  computePlayerUrl,
  isLikelyPfrPlayerId
} = context;

assert.strictEqual(typeof selectSportsRefSlugSource, 'function');
assert.strictEqual(typeof isLikelyPfrPlayerId, 'function');

assert.strictEqual(isLikelyPfrPlayerId('AbraJo00'), true);
assert.strictEqual(isLikelyPfrPlayerId('AndeRa21'), true);
assert.strictEqual(isLikelyPfrPlayerId('andrew-jones'), false);

const row = {
  'Player-additional': 'AbraJo00',
  slug: '',
  sportsref_url: '',
  sportsref_predicted_url: ''
};
const slugSource = selectSportsRefSlugSource(row, 'Abraham Jones');
assert.strictEqual(slugSource, 'Abraham Jones');
assert.strictEqual(normalizeSlugBase(slugSource), 'abraham-jones');
assert.strictEqual(
  computePlayerUrl(normalizeSlugBase(slugSource), 1),
  'https://www.sports-reference.com/cfb/players/abraham-jones-1.html'
);

const urlSource = selectSportsRefSlugSource(
  { sportsref_predicted_url: 'https://www.sports-reference.com/cfb/players/andre-jones-3.html' },
  'Andre Jones'
);
assert.strictEqual(normalizeSlugBase(urlSource), 'andre-jones');

console.log('slug selection tests passed');
