"""Layout of the verification layer.

The engine package must import cleanly and must not pull in the
DHSVM-specific modules: the ``dhsvm_*`` check modules and the contract
folder.  That separation is what lets a later step add other models.

Run with:  python -m pytest tests/verify -q      (from the repository root)
"""
import ast
import os

import ww_dhsvm.verify as V

VERIFY_DIR = os.path.dirname(V.__file__)


def test_verify_package_is_part_of_ww_dhsvm():
    assert V.__name__ == 'ww_dhsvm.verify'
    assert os.path.basename(os.path.dirname(VERIFY_DIR)) == 'ww_dhsvm'


def test_package_init_imports_no_dhsvm_rules():
    with open(os.path.join(VERIFY_DIR, '__init__.py')) as f:
        tree = ast.parse(f.read())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported += ['%s.%s' % (node.module or '', alias.name)
                         for alias in node.names]
    specific = [name for name in imported
                if any(part.startswith('dhsvm_') or part == 'contracts'
                       for part in name.split('.'))]
    assert specific == []


def test_dhsvm_contract_folder_has_a_readme():
    assert os.path.isfile(os.path.join(VERIFY_DIR, 'contracts', 'dhsvm',
                                       'README.md'))
