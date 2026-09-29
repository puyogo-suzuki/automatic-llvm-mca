#!/usr/bin/env python3
"""Compare an Arm Software Optimization Guide (SOG) against a scheduling model.

Input
  --sog  Text of the SOG, made with `pdftotext -layout <guide>.pdf <guide>.txt`.
         The SOGs themselves are NOT part of this repository.
  --csv  Output of `mca-insts-info --mtriple aarch64-linux-gnu --mcpu <cpu>
         --format csv` for the CPU the guide describes.

For every AArch64 instruction-table row of the SOG (group, mnemonics, execution
latency, execution throughput, utilized pipelines) the script looks up the model's
opcodes with those mnemonics and checks latency, throughput and the pipe set.
An opcode is reported only if it matches NONE of the SOG rows that name its
mnemonic, so ambiguity about which row an operand form belongs to (D-form vs
Q-form, element size, ...) does not produce a report by itself.

Reading the report
  * `diff=` says which of Latency / Throughput / Pipes disagree with the closest row.
  * The report contains false positives: rows the SOG itself prints inconsistently,
    instruction forms that are not distinguishable by mnemonic (scalar-SIMD forms vs
    GPR forms, F16 forms), and pipe sets where a group is listed together with one
    of its members.  Treat it as a triage list, not a verdict.
  * It looks at the model tables only.  Corrections applied on top of the model in
    facile.cpp (e.g. the A78 register-offset / M0-occupancy fixes) are not visible.
  * Opcodes whose mnemonic no SOG row names, and SOG rows whose mnemonic maps to no
    opcode (aliases such as CMP, MOV) are listed separately.

Usage
  tools/sog_compare.py --sog tmp/x1sog.txt --csv x1.csv [--model v1|n1|n2]

--model selects the resource-name prefix and group structure and is detected from
the CSV when omitted (V1Unit* -> v1, N2Unit* -> n2, N1Unit* -> n1).
"""
import argparse
import collections
import csv
import re
import signal
import sys

# ---------------------------------------------------------------------------
# Resource structure of the borrowed models.  Groups map to the units they
# contain; a resource whose unit set strictly contains another listed resource's
# unit set is dropped when reducing a model resource list to "pipes".
# ---------------------------------------------------------------------------
MODELS = {
    'n1': ('N1Unit', {
        'B': {'B'}, 'S': {'S'}, 'M': {'M'}, 'L': {'L'}, 'D': {'D'},
        'V0': {'V0'}, 'V1': {'V1'}, 'Flg': {'Flg'},
        'I': {'S', 'M'}, 'V': {'V0', 'V1'}}),
    'n2': ('N2Unit', {
        'B': {'B'}, 'S': {'S'}, 'M0': {'M0'}, 'M1': {'M1'}, 'L01': {'L01'},
        'L2': {'L2'}, 'D': {'D'}, 'V0': {'V0'}, 'V1': {'V1'}, 'Flg': {'Flg'},
        'I': {'S', 'M0', 'M1'}, 'M': {'M0', 'M1'}, 'L': {'L01', 'L2'},
        'V': {'V0', 'V1'}}),
    'v1': ('V1Unit', {
        'B': {'B'}, 'S': {'S'}, 'M0': {'M0'}, 'M1': {'M1'}, 'L01': {'L01'},
        'L2': {'L2'}, 'D': {'D'}, 'V0': {'V0'}, 'V1': {'V1'}, 'V2': {'V2'},
        'V3': {'V3'}, 'Flg': {'Flg'},
        'I': {'S', 'M0', 'M1'}, 'M': {'M0', 'M1'}, 'L': {'L01', 'L2'},
        'V': {'V0', 'V1', 'V2', 'V3'}, 'V01': {'V0', 'V1'},
        'V02': {'V0', 'V2'}, 'V13': {'V1', 'V3'}}),
}

# Mnemonics the SOG uses for aliases -> the mnemonics of the underlying opcodes.
ALIAS = {
    'CMP': ['SUBS'], 'CMN': ['ADDS'], 'TST': ['ANDS'], 'NEG': ['SUB'], 'NEGS': ['SUBS'],
    'MVN': ['ORN'], 'MOV': ['ORR', 'MOVZ', 'MOVN', 'MOVK', 'ADD', 'INS', 'UMOV', 'DUP'],
    'MUL': ['MADD'], 'MNEG': ['MSUB'], 'SMULL': ['SMADDL'], 'SMNEGL': ['SMSUBL'],
    'UMULL': ['UMADDL'], 'UMNEGL': ['UMSUBL'],
    'LSL': ['LSLV', 'UBFM', 'SHL'], 'LSR': ['LSRV', 'UBFM', 'USHR'],
    'ASR': ['ASRV', 'SBFM', 'SSHR'], 'ROR': ['RORV', 'EXTR'],
    'CSET': ['CSINC'], 'CINC': ['CSINC'], 'CNEG': ['CSNEG'], 'CINV': ['CSINV'],
    'NOP': ['NOP', 'HINT'], 'SXTW': ['SBFM'], 'UXTB': ['UBFM'], 'UXTH': ['UBFM'],
    'SXTB': ['SBFM'], 'SXTH': ['SBFM'],
}

# ---------------------------------------------------------------------------
# SOG parsing
# ---------------------------------------------------------------------------
HEADER = re.compile(r'^\s*Instruction [Gg]roup\s+(AArch64|AArch32)')
TAIL = re.compile(
    r'^(?P<left>.*?)\s+(?P<lat>\+?\d+(?: to \d+)?(?: ?\(\d+\))?)'
    r'\s+(?P<tp>\d+(?:\.\d+)?(?:/\d+)?(?: to \d+(?:\.\d+)?(?:/\d+)?)?)'
    r'\s+(?P<pipes>[A-Z][A-Z0-9,. ]*?)(?:\s+(?P<note>-|[\d ,]+))?\s*$')
SKIP = re.compile(r'Copyright|Non-Confidential|Arm Confidential|^\s*Page\b|Arm® Cortex|'
                  r'^\s*Issue \d|^\s*Ve\b|PJDOC|Instruction char|^\s*$|latency\s+throughput|'
                  r'Pipelines\s*$')


def parse_sog(path):
    rows, in_a64, inscol, last, table = [], False, None, None, None
    for ln in open(path, encoding='utf-8', errors='replace').read().split('\n'):
        m = re.match(r'^Table (3-\d+)', ln)
        if m:
            table = m.group(1)
        h = HEADER.match(ln)
        if h:  # a header line switches AArch64 / AArch32 and fixes the instruction column
            in_a64, inscol, last = (h.group(1) == 'AArch64'), ln.index(h.group(1)), None
            continue
        if not in_a64 or inscol is None or SKIP.search(ln):
            continue
        if re.match(r'^\s*Notes?:', ln) or re.match(r'^\s+\d+\.\s', ln):
            in_a64 = False
            continue

        def split_at(line):
            for cand in range(inscol - 3, inscol + 2):
                if 1 <= cand < len(line) and line[cand - 1] == ' ' and line[cand] != ' ':
                    return cand
            return inscol

        t = TAIL.match(ln)
        if t:
            s = split_at(ln)
            last = dict(table=table, group=ln[:s].strip(), ins=ln[s:len(t.group('left'))].strip(),
                        lat=t.group('lat'), tp=t.group('tp'), pipes=t.group('pipes').strip())
            rows.append(last)
        elif last is not None:  # continuation line of the previous row
            s = split_at(ln)
            if ln[:s].strip():
                last['group'] += ' ' + ln[:s].strip()
            if ln[s:].strip():
                last['ins'] += ' ' + ln[s:].strip()
    return rows


def num(x):
    if '/' in x:
        a, b = x.split('/')
        return float(a) / float(b)
    return float(x)


def parse_lat(text):
    text = text.strip()
    if text.startswith('+'):
        return None  # "+1" (branch forms) is not a latency of its own
    m = re.match(r'(\d+)(?: to (\d+))?(?: ?\((\d+)\))?$', text)
    return int(m.group(1)), int(m.group(2) or m.group(1)), int(m.group(3)) if m.group(3) else None


def parse_tp(text):
    if ' to ' in text:
        a, b = text.split(' to ')
        return num(a), num(b)
    return num(text), num(text)


def sog_mnemonics(ins):
    ins = re.sub(r'\([^)]*\)', ' ', ins)
    out = set()
    for tok in re.split(r'[,\s]+', ins):
        if not tok or not re.match(r'^[A-Z][A-Z0-9{}|.]*$', tok):
            continue
        if '{S}' in tok:
            base = tok.replace('{S}', '')
            out.update((base, base + 'S'))
        else:
            out.add(tok)
    res = set()
    for t in out:
        res.add(t.lower())
        res.update(a.lower() for a in ALIAS.get(t, []))
    return res


# ---------------------------------------------------------------------------
# Model side
# ---------------------------------------------------------------------------
def model_pipes(resources, prefix, units):
    if resources.strip() in ('-', ''):
        return frozenset()
    names = [r.split(':')[0].replace(prefix, '') for r in resources.strip('[]').split()]
    names = [n for n in names if n != 'Flg']
    keep = [n for n in names
            if n in units and not any(o != n and o in units and units[o] < units[n] for o in names)]
    return frozenset(keep)


def is_vector_name(name):
    return bool(re.search(r'v\d+i\d+|v\d+f\d+|v1i64|v2i64|v8i8|v16i8|v4i16|v8i16|v2i32|v4i32|'
                          r'v2f|v4f|v8f|v1f|LD\dR?v|ST\dv|LD\d|ST\d|^(SHA|AES|PMULL|SM3|SM4|EOR3|BCAX|RAX1|XAR)',
                          name))


def keep_opcode(name):
    """SVE / SME / FP8 opcodes carry an underscore (X1/A78/A76 do not implement them)."""
    stripped = re.sub(r'_(shift|ns)$', '', name)
    return '_' not in stripped and not re.search(r'v\d+f8$', stripped)


def row_match(row, opc, prefix, units):
    """True, or a (lat_ok, tp_ok, pipes_ok) tuple."""
    lat, rthr = float(opc['Latency']), float(opc['RThroughput'])
    pl = parse_lat(row['lat'])
    if pl is None:
        return True
    lo, hi, alt = pl
    lat_ok = lo <= lat <= hi or (alt is not None and lat == alt)
    tlo, thi = parse_tp(row['tp'])
    tp_ok = rthr > 0 and tlo * 0.85 <= 1.0 / rthr <= thi * 1.18
    sog_pipes = frozenset(('L01' if x == 'L0' else x) for x in (p.strip() for p in row['pipes'].split(',')) if x)
    return lat_ok, tp_ok, model_pipes(opc['Resources'], prefix, units) == sog_pipes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sog', required=True, help='SOG text (pdftotext -layout)')
    ap.add_argument('--csv', required=True, help='mca-insts-info --format csv output')
    ap.add_argument('--model', choices=sorted(MODELS), help='resource naming (detected from the CSV if omitted)')
    args = ap.parse_args()

    opcodes = list(csv.DictReader(open(args.csv)))
    model = args.model
    if model is None:
        text = ' '.join(o['Resources'] for o in opcodes[:2000])
        model = next((m for m, (pfx, _) in MODELS.items() if pfx in text), None)
        if model is None:
            sys.exit('cannot detect the model from the CSV; pass --model')
    prefix, units = MODELS[model]

    rows = parse_sog(args.sog)
    by_mnem = collections.defaultdict(list)
    for o in opcodes:
        if keep_opcode(o['OpcodeName']):
            by_mnem[o['Mnemonic'].lower()].append(o)

    candidates = collections.defaultdict(list)  # opcode name -> [(row, opcode)]
    for row in rows:
        row_is_vector = bool(re.match(r'ASIMD|Crypto|AES|SIMD|Advanced', row['group']))
        for mn in sog_mnemonics(row['ins']):
            for o in by_mnem.get(mn, []):
                if is_vector_name(o['OpcodeName']) == row_is_vector:
                    candidates[o['OpcodeName']].append((row, o))

    report = collections.defaultdict(list)
    for name, lst in candidates.items():
        best, best_score = None, -1
        for row, o in lst:
            r = row_match(row, o, prefix, units)
            if r is True or all(r):
                best = None
                break
            if sum(r) > best_score:
                best, best_score = (row, r), sum(r)
        if best is not None:
            o = lst[0][1]
            report[(o['Mnemonic'], o['Latency'], o['RThroughput'], o['Resources'])].append((name, best))

    unmapped = [r for r in rows if not any(by_mnem.get(mn) for mn in sog_mnemonics(r['ins']))]
    print(f"model={model}  SOG rows={len(rows)}  opcodes considered={len(candidates)}  "
          f"mismatching opcode groups={len(report)}")
    print("== SOG rows with no opcode of that mnemonic (aliases etc.):")
    for r in unmapped:
        print(f"   {r['group'][:50]} | {r['ins'][:40]}")
    print("== mismatches: model spec vs the closest SOG row")
    for (mn, lat, rthr, res), lst in sorted(report.items(), key=lambda kv: kv[0][0]):
        row, r = lst[0][1]
        diff = ''.join(c for c, ok in zip('LTP', r) if not ok)
        print(f"{mn:9s} model(lat={lat}, rthr={rthr}, {res}) vs SOG[{row['group'][:40]}: "
              f"lat={row['lat']} tp={row['tp']} pipes={row['pipes']}] diff={diff} "
              f"n={len(lst)} e.g. {','.join(n for n, _ in lst[:3])}")


if __name__ == '__main__':
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # allow `| head`
    main()
