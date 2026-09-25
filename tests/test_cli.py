"""CLI subcommands smoke (stdlib argparse; no Typer required)."""
import json

from glue_lp.cli import build_parser, run as main


def test_build_parser_has_subcommands():
    p = build_parser()
    # argparse stores subparsers; just ensure parse works
    args = p.parse_args(["describe-config"])
    assert args.command == "describe-config"


def test_protocol_check_exit0(capsys):
    code = main(["protocol-check", "--seed", "7"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["valid"]["invariant_held"] is True
    assert out["leaky"]["leakage"] is True


def test_describe_config_json(capsys):
    code = main(["describe-config"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["TrainConfig"]["hidden"] == 32
    assert payload["datasets"]["cora"]["n_nodes"] == 2708


def test_export_summary_serie1(capsys):
    code = main(["export-summary", "serie1"])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert "E1_gcn_valid_auc" in summary


def test_verify_results_ok():
    code = main(["verify-results"])
    assert code == 0


def test_leakage_ladder_print():
    code = main(["leakage-ladder"])
    assert code == 0
