"""Text length, units, object order and value swaps."""
import torch


def conditions(group):
    if group == 'necessity':
        return [(rep, 'cm', False, mode) for rep in ('lexical', 'trained', 'random', 'mean', 'trim')
                for mode in ('true', 'swapped')] + [('absent', 'cm', False, 'true')]
    if group == 'robustness':
        return [(rep, unit, reverse, mode) for unit, reverse in (('mm', False), ('m', False), ('cm', True))
                for rep in ('lexical', 'trained') for mode in ('true', 'swapped')] + [
                    (rep, 'cm', False, 'true') for rep in ('lexical', 'trained')]
    raise ValueError('unknown fixed stress group')


def value_text(cm, unit):
    if isinstance(cm, bool) or not isinstance(cm, int) or cm <= 0:
        raise ValueError('positive integer quantized centimeters required')
    if unit == 'cm':
        return f' {cm} cm'
    if unit == 'mm':
        return f' {cm*10} mm'
    if unit == 'm':
        return f' {cm//100}.{cm%100:02d} m'
    raise ValueError('undeclared unit')


def parts_for(visible_ids, centimeters, *, unit, reverse, mode):
    if len(visible_ids) != 2 or len(set(visible_ids)) != 2 or len(centimeters) != 2:
        raise ValueError('two distinct visible IDs and measurements required')
    if mode not in ('true', 'swapped'):
        raise ValueError('unknown intervention')
    values = list(centimeters) if mode == 'true' else list(reversed(centimeters))
    entries = list(zip(visible_ids, values))
    if reverse:
        entries.reverse()
    (a, ca), (b, cb) = entries
    return [f' Measured metric radius: <OBJ{a:03d}> =', value_text(ca, unit),
            f'; <OBJ{b:03d}> =', value_text(cb, unit), '.']


def compress(embeddings, token_ids, *, representation, trained, random, decode):
    if representation == 'lexical':
        return embeddings
    if representation == 'trained':
        return trained(embeddings)
    if representation == 'random':
        return random(embeddings)
    if representation == 'mean':
        return embeddings.float().mean(0, keepdim=True).expand(2, -1).to(embeddings.dtype)
    if representation == 'trim':
        if len(token_ids) != len(embeddings) or len(token_ids) < 2 or decode([token_ids[0]]) != ' ':
            raise ValueError('trim requires exactly a standalone leading space token')


        return embeddings[1:]
    raise ValueError('unknown packet representation')
