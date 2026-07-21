import pytest

from bire_repro.cli import main


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as raised:
        main(["--help"])
    assert raised.value.code == 0
    assert "AF-FNO" in capsys.readouterr().out
