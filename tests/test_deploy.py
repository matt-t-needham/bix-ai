"""deploy.py queue contract + /deploys routes."""
import json

import pytest

import config
import deploy
import staging


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FS_ROOT", tmp_path.resolve())
    monkeypatch.setattr(config, "STAGING_DIR", tmp_path / "staging")
    monkeypatch.setattr(config, "DEPLOY_DIR", tmp_path / "deploy")
    return tmp_path


# ── enqueue ───────────────────────────────────────────────────────────────────

def test_enqueue_writes_queue_json(env):
    req = deploy.enqueue("deploy-staging", note="test run")
    path = config.DEPLOY_DIR / "queue" / f"{req['id']}.json"
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk == req
    assert on_disk["action"] == "deploy-staging"
    assert on_disk["record_ids"] == []
    assert on_disk["requested_by"] == "human"
    # no stray .tmp left behind (atomic write)
    assert list((config.DEPLOY_DIR / "queue").glob("*.tmp")) == []


def test_enqueue_invalid_action(env):
    with pytest.raises(ValueError):
        deploy.enqueue("rm-rf-everything")
    assert deploy.list_all() == []


def test_promote_requires_record_ids(env):
    with pytest.raises(ValueError, match="at least one"):
        deploy.enqueue("promote")


def test_promote_requires_existing_approved_self_change(env):
    with pytest.raises(ValueError, match="no such"):
        deploy.enqueue("promote", record_ids=["doesnotexist"])

    # pending self-change → refused
    rec = staging.create(str(env / "bix-ai" / "helpers.py"), "# x")
    with pytest.raises(ValueError, match="not approved"):
        deploy.enqueue("promote", record_ids=[rec["id"]])

    # approved but NOT a self-change → refused
    other = staging.create(str(env / "bix-blog" / "post.md"), "hi")
    staging.approve(other["id"])
    with pytest.raises(ValueError, match="not a bix-ai self-change"):
        deploy.enqueue("promote", record_ids=[other["id"]])

    # approved self-change → accepted
    staging.approve(rec["id"])
    req = deploy.enqueue("promote", record_ids=[rec["id"]])
    assert req["record_ids"] == [rec["id"]]


# ── get / result merging ──────────────────────────────────────────────────────

def test_get_defaults_to_queued(env):
    req = deploy.enqueue("deploy-staging")
    dep = deploy.get(req["id"])
    assert dep["status"] == "queued"
    assert dep["action"] == "deploy-staging"


def test_get_merges_runner_result(env):
    req = deploy.enqueue("deploy-staging")
    # simulate the runner: claim into processing/, write a result
    q = config.DEPLOY_DIR / "queue" / f"{req['id']}.json"
    p = config.DEPLOY_DIR / "processing" / f"{req['id']}.json"
    p.parent.mkdir(parents=True)
    q.rename(p)
    (config.DEPLOY_DIR / "results").mkdir(parents=True)
    (config.DEPLOY_DIR / "results" / f"{req['id']}.json").write_text(json.dumps({
        "id": req["id"], "status": "success", "exit_code": 0, "git_sha_after": "abc1234",
    }))
    dep = deploy.get(req["id"])
    assert dep["status"] == "success"
    assert dep["exit_code"] == 0
    assert dep["action"] == "deploy-staging"     # request fields survive the merge
    assert dep["git_sha_after"] == "abc1234"


def test_get_unknown_id(env):
    assert deploy.get("nope") is None
    assert deploy.get("../../etc/passwd") is None


# ── list_all ordering ─────────────────────────────────────────────────────────

def test_list_all_active_first(env):
    done = deploy.enqueue("deploy-staging", note="old")
    (config.DEPLOY_DIR / "results").mkdir(parents=True)
    (config.DEPLOY_DIR / "results" / f"{done['id']}.json").write_text(
        json.dumps({"id": done["id"], "status": "success"}))
    active = deploy.enqueue("deploy-staging", note="new")
    deps = deploy.list_all()
    assert [d["id"] for d in deps] == [active["id"], done["id"]]


# ── read_log ──────────────────────────────────────────────────────────────────

def test_read_log_tail(env):
    req = deploy.enqueue("deploy-staging")
    logs = config.DEPLOY_DIR / "logs"
    logs.mkdir(parents=True)
    (logs / f"{req['id']}.log").write_text("\n".join(f"line{i}" for i in range(500)))
    tail = deploy.read_log(req["id"], lines=100)
    assert tail.splitlines()[0] == "line400"
    assert tail.splitlines()[-1] == "line499"
    assert deploy.read_log("missing") == ""


# ── Routes ────────────────────────────────────────────────────────────────────

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("forge")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(env):
    import main
    return TestClient(main.app)


def test_deploy_count_route_order(client):
    # /deploys/count must not be captured by /deploys/{dep_id}.
    r = client.get("/deploys/count")
    assert r.status_code == 200
    assert r.json() == {"active": 0, "total": 0}


def test_deploys_list_page(client):
    assert "No deploys yet" in client.get("/deploys").text


def test_enqueue_route_and_detail(client):
    r = client.post("/deploys/enqueue", data={"action": "deploy-staging", "note": "hi"},
                    follow_redirects=False)
    assert r.status_code == 303
    dep_id = r.headers["location"].rsplit("/", 1)[-1]
    page = client.get(f"/deploys/{dep_id}").text
    assert "deploy-staging" in page and "queued" in page
    assert 'http-equiv="refresh"' in page          # active → meta-refresh
    assert client.get("/deploys/count").json() == {"active": 1, "total": 1}


def test_enqueue_route_invalid_action(client):
    r = client.post("/deploys/enqueue", data={"action": "nope"})
    assert r.status_code == 400


def test_enqueue_route_405_on_staging_role(client, monkeypatch):
    monkeypatch.setattr(config, "BIX_ROLE", "staging")
    r = client.post("/deploys/enqueue", data={"action": "deploy-staging"})
    assert r.status_code == 405


def test_rollback_button_on_successful_promote(client, env):
    rec = staging.create(str(env / "bix-ai" / "helpers.py"), "# x")
    staging.approve(rec["id"])
    req = deploy.enqueue("promote", record_ids=[rec["id"]])
    (config.DEPLOY_DIR / "results").mkdir(parents=True, exist_ok=True)
    (config.DEPLOY_DIR / "results" / f"{req['id']}.json").write_text(
        json.dumps({"id": req["id"], "status": "success"}))
    page = client.get(f"/deploys/{req['id']}").text
    assert "Roll back this promote" in page
    assert 'http-equiv="refresh"' not in page      # finished → no auto-refresh


def test_summary_tag_and_text_rendered(client, env):
    # Runner-written local-model summary fields surface in both views: the
    # tag as a chip in the list and header, the description on the detail.
    req = deploy.enqueue("deploy-staging")
    (config.DEPLOY_DIR / "results").mkdir(parents=True, exist_ok=True)
    (config.DEPLOY_DIR / "results" / f"{req['id']}.json").write_text(json.dumps({
        "id": req["id"], "status": "failed", "exit_code": 1,
        "summary_tag": "pytest gate failure",
        "summary_text": "test_detail_escapes_content failed in the Docker test stage.",
        "summary_model": "gemma4:26b",
    }))
    listing = client.get("/deploys").text
    assert "pytest gate failure" in listing
    detail = client.get(f"/deploys/{req['id']}").text
    assert "pytest gate failure" in detail
    assert "test_detail_escapes_content failed" in detail
    assert "gemma4:26b · local summary" in detail


def test_no_summary_renders_clean(client, env):
    # Absent summary fields (Ollama down, old results) must not leave stray
    # chips or an empty summary block.
    req = deploy.enqueue("deploy-staging")
    (config.DEPLOY_DIR / "results").mkdir(parents=True, exist_ok=True)
    (config.DEPLOY_DIR / "results" / f"{req['id']}.json").write_text(
        json.dumps({"id": req["id"], "status": "success"}))
    detail = client.get(f"/deploys/{req['id']}").text
    assert "local summary" not in detail
    assert 'badge tag' not in detail


def test_ids_are_full_length_uuids(env):
    req = deploy.enqueue("deploy-staging")
    assert len(req["id"]) == 32
    rec = staging.create(str(env / "x.md"), "hi", "gemma4:26b")
    assert len(rec["id"]) == 32


def test_implemented_by_models_from_note_and_records(client, env):
    rec = staging.create(str(env / "x.md"), "hi", "gemma4:26b")
    staging.update_content(rec["id"], "hi2", "claude:claude-sonnet-4-6")
    # staging page's deploy button passes the record via the note only
    req = deploy.enqueue("deploy-staging", note=f"staging record {rec['id']}")
    listing = client.get("/deploys").text
    assert "implemented by gemma4:26b, claude-sonnet-4-6" in listing
    assert f"ID {req['id']}" in listing
    detail = client.get(f"/deploys/{req['id']}").text
    assert "implemented by" in detail
    assert "gemma4:26b, claude-sonnet-4-6" in detail
    assert f"ID {req['id']}" in detail


def test_stage_write_records_acting_model(env):
    import asyncio
    import re as _re
    import tools
    out = asyncio.run(tools._tool_stage_write({
        "target_path": str(env / "y.md"), "content": "hi",
        "_acting_model": "gemma4:26b",
    }))
    assert "Staged for review" in out
    rec_id = _re.search(r"id=([0-9a-f]+)", out).group(1)
    assert staging.get(rec_id)["proposed_by"] == "gemma4:26b"


def test_read_deploy_tool(env):
    import asyncio
    import tools
    req = deploy.enqueue("deploy-staging", note="staging record abc")
    (config.DEPLOY_DIR / "results").mkdir(parents=True, exist_ok=True)
    (config.DEPLOY_DIR / "results" / f"{req['id']}.json").write_text(json.dumps({
        "id": req["id"], "status": "failed", "exit_code": 1,
        "summary_tag": "pytest gate failure",
    }))
    (config.DEPLOY_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (config.DEPLOY_DIR / "logs" / f"{req['id']}.log").write_text(
        "FAILED tests/test_x.py::test_y - AssertionError\n=== failed exit=1 ===")
    out = asyncio.run(tools._tool_read_deploy({"deploy_id": req["id"]}))
    assert "status: failed" in out
    assert "pytest gate failure" in out
    assert "FAILED tests/test_x.py::test_y" in out          # runner log tail included
    # unique prefix resolves; unknown id doesn't
    assert "status: failed" in asyncio.run(tools._tool_read_deploy({"deploy_id": req["id"][:8]}))
    assert "No deploy found" in asyncio.run(tools._tool_read_deploy({"deploy_id": "ffffffff"}))


def test_list_deploys_tool(env):
    import asyncio
    import tools
    assert asyncio.run(tools._tool_list_deploys({})) == "No deploys."
    req = deploy.enqueue("deploy-staging", note="hello")
    out = asyncio.run(tools._tool_list_deploys({}))
    assert req["id"] in out and "queued" in out and "hello" in out
