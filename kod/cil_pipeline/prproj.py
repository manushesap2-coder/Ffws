"""Print the edit of a Premiere Pro project (.prproj is gzipped XML): per sequence, each track item with timeline
start/end, source file and in/out, speed and motion params (Scale, Rotation, Position, Opacity, Level), plus markers.
Usage: python prproj.py project.prproj [out.json]"""
import gzip, json, sys
import xml.etree.ElementTree as ET

TPS = 254016000000                      # Premiere ticks per second
PARAMS = ('Position', 'Scale', 'Rotation', 'Opacity', 'Level')


def main():
    root = ET.fromstring(gzip.open(sys.argv[1]).read())
    by_id, by_uid = {}, {}
    for el in root.iter():
        if 'ObjectID' in el.attrib:
            by_id[el.attrib['ObjectID']] = el
        if 'ObjectUID' in el.attrib:
            by_uid[el.attrib['ObjectUID']] = el

    def ref(el):
        if el is None:
            return None
        if 'ObjectRef' in el.attrib:
            return by_id.get(el.attrib['ObjectRef'])
        return by_uid.get(el.attrib.get('ObjectURef'))

    def media_name(clip):
        src = ref(clip.find('./Clip/Source'))
        media = ref(src.find('.//MediaSource/Media')) if src is not None else None
        if media is None:
            return '?'
        path = media.findtext('FilePath') or media.findtext('ActualMediaFilePath') or '?'
        return path.replace('\\', '/').split('/')[-1]

    def params_of(cti):
        out = {}
        comps = ref(cti.find('./ComponentOwner/Components'))
        if comps is None:
            return out
        for c in comps.findall('.//Component'):
            comp = ref(c)
            for p in (comp.findall('.//Param') if comp is not None else []):
                prm = ref(p)
                name = prm.findtext('Name') if prm is not None else None
                if name in PARAMS:
                    val = prm.findtext('StartKeyframe') or prm.findtext('CurrentValue') or ''
                    out[name] = (val.split(',')[1] if ',' in val else val) + (' (keyframed)' if prm.findtext('Keyframes') else '')
        return out

    rows = []
    for seq in root.iter('Sequence'):
        for tg in seq.findall('./TrackGroups/TrackGroup'):
            grp = ref(tg.find('Second'))
            if grp is None:
                continue
            for tr in grp.findall('./TrackGroup/Tracks/Track'):
                track = ref(tr)
                if track is None:
                    continue
                for it in track.findall('./ClipTrack/ClipItems/TrackItems/TrackItem'):
                    item = ref(it)
                    cti = item.find('ClipTrackItem') if item is not None else None
                    sub = ref(cti.find('SubClip')) if cti is not None else None
                    clip = ref(sub.find('Clip')) if sub is not None else None
                    if clip is None:
                        continue
                    out = clip.findtext('./Clip/OutPoint')
                    rows.append(dict(sequence=seq.findtext('Name') or '?', kind=grp.tag.replace('TrackGroup', ''),
                                     track=tr.attrib.get('Index', '0'),
                                     start=int(cti.findtext('./TrackItem/Start') or 0) / TPS,
                                     end=int(cti.findtext('./TrackItem/End') or 0) / TPS,
                                     file=media_name(clip), src_in=int(clip.findtext('./Clip/InPoint') or 0) / TPS,
                                     src_out=int(out) / TPS if out else None,
                                     speed=float(clip.findtext('./Clip/PlaybackSpeed') or 1), params=params_of(cti)))
    for r in rows:
        p = ' '.join(f'{k}={v}' for k, v in r['params'].items())
        src_out = f"{r['src_out']:.2f}" if r['src_out'] is not None else '?'
        print(f"{r['sequence'][:22]:22} {r['kind'][:5]:5} T{r['track']} {r['start']:7.2f}-{r['end']:7.2f} ({r['end'] - r['start']:5.2f}s) "
              f"{r['file'][:40]:40} src {r['src_in']:.2f}-{src_out} x{r['speed']:g} {p}")
    markers = [el for el in root.iter() if 'Marker' in el.tag and any((c.text or '').strip() for c in el)]
    for el in markers[:40]:
        print('marker', el.tag, {c.tag: c.text.strip()[:80] for c in el if (c.text or '').strip()})
    if len(sys.argv) > 2:
        json.dump(rows, open(sys.argv[2], 'w'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
