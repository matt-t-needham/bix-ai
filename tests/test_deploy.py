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
