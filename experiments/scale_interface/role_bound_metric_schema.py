"""Conversion of object measurements to centimeters."""
from dataclasses import dataclass
from fractions import Fraction
import re


@dataclass(frozen=True)
class MetricRecord:
    object_id: int
    value: str
    unit: str
    valid: bool = True


def _check_id(value):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 999:
        raise ValueError('explicit native three-digit object ID required')


def canonical_parts(records, queried_ids):
    if len(records) != 2 or len(queried_ids) != 2:
        raise ValueError('this boundary supports exactly two queried records')
    for obj in queried_ids:
        _check_id(obj)
    if len(set(queried_ids)) != 2:
        raise ValueError('queried IDs must differ')
    values = {}
    for record in records:
        if not isinstance(record, MetricRecord):
            raise TypeError('typed MetricRecord required')
        _check_id(record.object_id)
        if record.object_id in values or record.valid is not True:
            raise ValueError('duplicate ID or invalid measurement')


        if (not isinstance(record.value, str) or len(record.value) > 40
                or re.fullmatch(r'[0-9]+(?:\.[0-9]+)?', record.value) is None):
            raise ValueError('bounded positive plain-decimal string required')
        if record.unit not in ('cm', 'mm', 'm'):
            raise ValueError('unsupported physical unit')
        factor = {'cm': Fraction(1), 'mm': Fraction(1, 10), 'm': Fraction(100)}[record.unit]
        cm = Fraction(record.value) * factor
        if cm <= 0 or cm.denominator != 1:
            raise ValueError('positive exact centimeter-grid quantity required; no implicit rounding')
        values[record.object_id] = cm.numerator
    if set(values) != set(queried_ids):
        raise ValueError('missing queried measurement or unrequested object')
    a, b = queried_ids
    return [f' Measured metric radius: <OBJ{a:03d}> =', f' {values[a]} cm',
            f'; <OBJ{b:03d}> =', f' {values[b]} cm', '.']
