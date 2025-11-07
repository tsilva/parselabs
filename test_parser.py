#!/usr/bin/env python3
"""Test the plain text format parser."""

from extraction import _parse_plain_text_format
import json

# Test cases from the error messages
test_cases = [
    "QUÍMICA CLÍNICA - URINA - URINA-TIPO II - Cor: AMARELA",
    "QUÍMICA CLÍNICA - URINA - URINA-TIPO II - pH: 7,0 (reference_range: 5,0-8,0)",
    "QUÍMICA CLÍNICA - URINA - URINA-TIPO II - Densidade a 15°C: 1,015 (reference_range: 1,005 - 1,035)",
    "QUÍMICA CLÍNICA - URINA - ELEMENTOS ANORMAIS - Proteínas: NÃO CONTÉM",
    "QUÍMICA CLÍNICA - URINA - ELEMENTOS ANORMAIS - Glicose: NÃO CONTÉM",
    "QUÍMICA CLÍNICA - URINA - ELEMENTOS ANORMAIS - Corpos cetónicos: NÃO CONTÉM",
    "QUÍMICA CLÍNICA - URINA - ELEMENTOS ANORMAIS - Bilirrubina: NÃO CONTÉM",
    "QUÍMICA CLÍNICA - URINA - ELEMENTOS ANORMAIS - Nitritos: NÃO CONTÉM",
    "QUÍMICA CLÍNICA - URINA - ELEMENTOS ANORMAIS - Sangue: NÃO CONTÉM",
    "QUÍMICA CLÍNICA - URINA - ELEMENTOS ANORMAIS - Urobilinogenio: NORMAL",
    "QUÍMICA CLÍNICA - URINA - EXAME MICROSCÓPICO DO SEDIMENTO - Células epiteliais: 1 - 2/CAMPO",
    "QUÍMICA CLÍNICA - URINA - EXAME MICROSCÓPICO DO SEDIMENTO - Leucócitos: 0 - 1/CAMPO",
    "QUÍMICA CLÍNICA - FEZES - EXAME PARASITOLÓGICO: NEGATIVO"
]

print("Testing plain text format parser:\n")
for i, test_str in enumerate(test_cases):
    print(f"Test {i+1}:")
    print(f"  Input: {test_str}")
    result = _parse_plain_text_format(test_str)
    if result:
        print(f"  ✓ Parsed successfully:")
        print(f"    test_name: {result['test_name']}")
        print(f"    value: {result['value']}")
        print(f"    unit: {result['unit']}")
        print(f"    reference_range: {result['reference_range']}")
        print(f"    comments: {result['comments']}")
    else:
        print(f"  ✗ Failed to parse")
    print()
