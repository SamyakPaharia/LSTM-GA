// Squads. All player names are invented. Row: name | role | bat | bowl | style | hand
// role: bat, wk, ar, bowl. style: RF RFM RM LF LFM (pace), OS LS SLA (spin), - (does not bowl).

const RAW = {
  IND: {
    name: 'India', shirt: 0x2b8ae6, trim: 0xff8a1e, pants: 0x2477c9,
    squad: [
      'Arnav Kulshreshth|bat|84|8|-|R', 'Dev Ranawat|bat|80|30|OS|L', 'Ishan Vartak|bat|91|25|RM|R',
      'Rudra Bhosle|bat|83|10|-|R', 'Kunal Mehrotra|wk|78|0|-|R', 'Tejas Nambiar|ar|73|72|RFM|R',
      'Samar Chettri|ar|66|77|SLA|L', 'Veer Solanke|bowl|41|80|OS|R', 'Aditya Phogat|bowl|30|84|RF|R',
      'Nikhil Bordoloi|bowl|22|87|RF|R', 'Yash Tandel|bowl|18|83|LS|R',
    ],
  },
  AUS: {
    name: 'Australia', shirt: 0xf2c230, trim: 0x0b6b3a, pants: 0xe3b21f,
    squad: [
      'Callum Hartigan|bat|85|10|-|R', 'Jed Marlowe|bat|82|12|-|L', 'Lachlan Prewitt|bat|89|35|LS|R',
      'Toby Wexford|bat|80|20|OS|R', 'Angus Kinsella|wk|77|0|-|L', 'Mitch Tolhurst|ar|72|74|RFM|R',
      'Rory Baines|ar|64|76|OS|R', 'Darcy Ellwood|bowl|38|86|LF|L', 'Hamish Corrigan|bowl|28|87|RF|R',
      'Nate Fairley|bowl|24|82|RFM|R', 'Ollie Standish|bowl|16|83|LS|R',
    ],
  },
  ENG: {
    name: 'England', shirt: 0xc62836, trim: 0x16264f, pants: 0xb5202d,
    squad: [
      'Freddie Ashdown|bat|83|10|-|R', 'Giles Penhallow|bat|81|15|-|L', 'Marcus Thorley|bat|88|30|OS|R',
      'Ned Carraway|bat|82|22|RM|R', 'Alfie Brackwell|wk|80|0|-|R', 'Jonty Sedgwick|ar|75|73|RFM|R',
      'Kit Hollis|ar|65|75|OS|L', 'Barney Oldcastle|bowl|42|79|RFM|R', 'Lewis Garside|bowl|26|86|RF|R',
      'Tom Hexham|bowl|22|83|LFM|L', 'Rafe Winstanley|bowl|20|82|LS|R',
    ],
  },
  PAK: {
    name: 'Pakistan', shirt: 0x2aa84a, trim: 0xe9f5c9, pants: 0x23913f,
    squad: [
      'Zayan Qazi|bat|82|8|-|R', 'Faris Durrani|bat|80|14|-|L', 'Arham Lodhi|bat|90|20|OS|R',
      'Saad Mirani|bat|79|12|-|R', 'Rayan Kakar|wk|81|0|-|R', 'Daniyal Tareen|ar|68|78|LS|R',
      'Ibrahim Soomro|ar|66|74|SLA|L', 'Usman Jatoi|bowl|32|88|RF|R', 'Taimur Baloch|bowl|28|87|LF|L',
      'Hasnain Awan|bowl|24|83|RFM|R', 'Moiz Chaudhry|bowl|18|80|OS|R',
    ],
  },
  RSA: {
    name: 'South Africa', shirt: 0x0d5c46, trim: 0xffd24a, pants: 0x0b4f3c,
    squad: [
      'Ruan Lotter|bat|84|10|-|R', 'Thabo Mokwena|bat|81|10|-|L', 'Dirk Vosloo|bat|87|28|RM|R',
      'Sipho Ndlela|bat|80|18|OS|R', 'Henk Retief|wk|78|0|-|R', 'Wikus Steenkamp|ar|74|75|RFM|R',
      'Lwazi Mthembu|ar|62|78|SLA|L', 'Brandt Fourie|bowl|34|88|RF|R', 'Andile Sokhela|bowl|27|84|RFM|R',
      'Ruben Coetzer|bowl|22|83|LF|L', 'Zain Abrahams|bowl|18|81|LS|R',
    ],
  },
  NZL: {
    name: 'New Zealand', shirt: 0x2b2f36, trim: 0x9fe3e0, pants: 0x24272d,
    squad: [
      'Hugo Tresidder|bat|82|10|-|R', 'Finn Atkins|bat|83|14|-|L', 'Wiremu Paraone|bat|89|26|OS|R',
      'Caleb Strang|bat|80|16|-|R', 'Jesse Lowry|wk|77|0|-|R', 'Tane Ruatapu|ar|73|72|RM|R',
      'Eli Carrick|ar|64|77|SLA|L', 'Mason Greig|bowl|40|82|RFM|R', 'Blake Sorensen|bowl|26|86|RF|R',
      'Reuben Hart|bowl|22|85|LFM|L', 'Isaac Mulholland|bowl|19|80|LS|R',
    ],
  },
  WIN: {
    name: 'West Indies', shirt: 0x7b1e3a, trim: 0xf2c230, pants: 0x6d1a33,
    squad: [
      'Jamal Pemberton|bat|83|10|-|R', 'Tyrell Lashley|bat|82|16|-|L', 'Marlon Sandiford|bat|86|20|OS|R',
      'Akeem Drayton|bat|84|24|RM|R', 'Dario Blackman|wk|76|0|-|L', 'Javon Springer|ar|77|74|RFM|R',
      'Shaquille Mapp|ar|63|76|OS|R', 'Renaldo Cumberbatch|bowl|36|85|RF|R', 'Kareem Yearwood|bowl|28|86|RF|R',
      'Jomel Forde|bowl|22|82|LFM|L', 'Devon Alleyne|bowl|18|81|LS|R',
    ],
  },
  SRL: {
    name: 'Sri Lanka', shirt: 0x2440b5, trim: 0xf5c518, pants: 0x1f389e,
    squad: [
      'Kavindu Ratnasiri|bat|82|10|-|R', 'Pasan Weerakkody|bat|80|22|OS|L', 'Ravindu Jayasekara|bat|87|18|-|R',
      'Lahiru Abeykoon|bat|81|26|OS|R', 'Sachith Palliyaguru|wk|77|0|-|R', 'Isuru Wanniarachchi|ar|71|73|RFM|R',
      'Minod Hettige|ar|65|79|OS|R', 'Tharaka Illangakoon|bowl|36|84|LS|R', 'Kasun Edirisooriya|bowl|27|85|RF|R',
      'Dulaj Senaratne|bowl|22|82|LFM|L', 'Amila Kodituwakku|bowl|18|81|SLA|L',
    ],
  },
};

function parse(id, raw) {
  const players = raw.squad.map((row, idx) => {
    const [name, role, bat, bowl, style, hand] = row.split('|');
    return { idx, name, short: shortName(name), role, bat: +bat, bowl: +bowl, style: style === '-' ? '' : style, hand };
  });
  return {
    id, name: raw.name, shirt: raw.shirt, trim: raw.trim, pants: raw.pants, players,
    keeper: players.findIndex((p) => p.role === 'wk'),
  };
}

function shortName(name) {
  const parts = name.split(' ');
  return parts[0][0] + ' ' + parts.slice(1).join(' ');
}

export const TEAMS = Object.fromEntries(Object.entries(RAW).map(([id, raw]) => [id, parse(id, raw)]));
export const TEAM_IDS = Object.keys(TEAMS);

export function getTeam(id) {
  return TEAMS[id] || TEAMS.IND;
}

export function hex(color) {
  return '#' + color.toString(16).padStart(6, '0');
}
