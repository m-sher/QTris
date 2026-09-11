#include <assert.h>
#include "../tetrisenv/TetrisEnv/b2b_search.c"

static void real_statistics(void) {
    MNode node = {0};
    MTree* tree = calloc(1, sizeof(MTree));
    MConfig cfg = {.gamma = 1};
    tree->qmin = 1e30f; tree->qmax = -1e30f;
    PathEntry path[] = {{&node, 0}};
    for (int k = 0; k < 8; k++) mtree_apply_vloss(path, 1, 1);
    assert(node.N[0] == 0 && node.virtual_visits[0] == 8);
    for (int k = 0; k < 8; k++) {
        mtree_revert_vloss(path, 1, 1);
        mtree_backup(tree, &cfg, path, 1, 0.2f);
        assert(fabsf(node.Q[0] - 0.2f) < 1e-6f);
        assert(tree->qmin > 0.19999f && tree->qmax < 0.20001f);
    }
    assert(node.N[0] == 8 && node.virtual_visits[0] == 0);
    assert(tree->completed == 8);
    free(tree);
}

static void gate_boundaries(void) {
    MRisk risks = {0};
    MNode node = {.n_legal = 3, .legal = {0, 1, 2}, .risk = &risks};
    MConfig cfg = {.risk_gate = 1, .risk_threshold = .10f, .risk_margin = .05f};
    uint8_t eligible[MCAP];
    risks.breaks[2] = true;
    risks.prediction[0][23] = .10f;
    risks.prediction[1][23] = .30f;
    risks.prediction[2][23] = .0f;
    node.prior[2] = 1;
    node.N[2] = 1000;
    node.W[2] = 1e6f;
    node.Q[2] = 1000;
    mcts_gate(&node, &cfg, eligible);
    assert(eligible[0] && !eligible[1] && !eligible[2]);
    assert(mcts_select(&node, &cfg, 0, 1000) == 0);
    node.N[2] = 0;
    risks.prediction[0][23] = .25f;
    risks.prediction[2][23] = .21f;
    mcts_gate(&node, &cfg, eligible);
    assert(eligible[0] && !eligible[2]);
    risks.prediction[2][23] = .20f;
    mcts_gate(&node, &cfg, eligible);
    assert(!eligible[0] && eligible[2]);
    risks.immediate_death[2] = true;
    mcts_gate(&node, &cfg, eligible);
    assert(eligible[0] && !eligible[2]);
    risks.immediate_death[2] = false;
    risks.breaks[0] = risks.breaks[1] = true;
    mcts_gate(&node, &cfg, eligible);
    assert(eligible[2]);
    risks.prediction[2][23] = .25f;
    mcts_gate(&node, &cfg, eligible);
    assert(eligible[0] && eligible[2]);
}

static void risk_backup(void) {
    MRisk a = {0}, b = {0}, c = {0};
    MNode root = {.risk = &a}, child = {.risk = &b};
    MNode leaf = {.risk = &c, .n_legal = 1, .legal = {0}};
    leaf.prior[0] = 1;
    root.child[0] = &child; child.child[0] = &leaf;
    MConfig cfg = {.gamma = .5f, .risk_gate = 1, .risk_threshold = .1f, .risk_margin = .05f};
    MTree* tree = calloc(1, sizeof(MTree));
    PathEntry path[] = {{&root, 0}, {&child, 0}};
    for (int h = 0; h < RISK_HORIZON; h++) c.prediction[0][h] = .01f * (h + 1);
    root.edge_reward[0] = root.edge_value[0] = .006f;
    child.edge_reward[0] = child.edge_value[0] = .012f;
    mtree_backup(tree, &cfg, path, 2, .04f);
    assert(a.sum[0][0] == 0 && a.sum[0][1] == 0);
    assert(fabsf(a.sum[0][23] - .22f) < 1e-6f);
    assert(fabsf(root.Q[0] - .022f) < 1e-6f);
    assert(fabsf(root.Wv[0] - .022f) < 1e-6f);
    b.immediate_death[0] = true;
    leaf.terminal = true;
    mtree_backup(tree, &cfg, path, 2, 0);
    assert(a.sum[0][0] == 0);
    assert(a.sum[0][1] == 1);
    for (int h = 1; h < RISK_HORIZON; h++)
        assert(mcts_action_risk(&root, 0, h) >= mcts_action_risk(&root, 0, h-1));
    free(tree);
}

static void search_admission_and_final_choice(void) {
    MRisk risks = {0};
    MNode node = {.n_legal = 3, .legal = {0, 1, 2}, .risk = &risks};
    MConfig cfg = {.risk_gate = 1, .risk_threshold = .10f, .risk_margin = .05f,
                   .c_puct = 1.5f, .fpu = .4f};
    risks.prediction[0][23] = .15f;
    risks.prediction[1][23] = .1501f;
    risks.prediction[2][23] = .1499f;
    risks.breaks[2] = true;
    node.prior[1] = 1;
    node.prior[2] = 1000;
    uint8_t eligible[MCAP];
    mcts_gate(&node, &cfg, eligible);
    assert(eligible[0] && !eligible[1] && !eligible[2]);
    assert(mcts_select(&node, &cfg, 0, 0) == 1);
    risks.immediate_death[1] = true;
    assert(mcts_select(&node, &cfg, 0, 0) == 0);
    risks.prediction[2][23] = .09f;
    assert(mcts_select(&node, &cfg, 0, 0) == 2);
    mcts_gate(&node, &cfg, eligible);
    assert(!eligible[0] && !eligible[1] && eligible[2]);
}

static void attack_credit_and_cancellation(void) {
    b2b_init_pieces();
    MConfig cfg = {.board_height = 40, .max_holes = -1, .risk_gate = 1};
    int plain_desc[] = {0, 0, 0, 38, 0};
    MState state = {.active = PIECE_I, .b2b = 6, .combo = -1, .qlen = 1,
                    .queue = {PIECE_I}, .gcnt = 1};
    state.gq[0] = (GarbEntry){.rows = 10, .col = 3, .timer = 1};
    state.board[39] = 0x3f0;
    state.board[38] = 0x200;
    MState original = state;
    bool terminal = false, plain = false;
    float credit = -1;
    float raw = mcts_apply_step(&state, &cfg, plain_desc, &terminal, &credit, &plain);
    assert(raw == 7 && credit == 0 && state.b2b == -1);
    assert(garb_total(state.gq, state.gcnt) == 3);
    state = original;
    plain_desc[4] = SPIN_ALL_MINI;
    raw = mcts_apply_step(&state, &cfg, plain_desc, &terminal, &credit, &plain);
    assert(raw == 1 && credit == raw && state.b2b == 7);
    assert(garb_total(state.gq, state.gcnt) == 9);
    state = original;
    state.board[38] = 0;
    plain_desc[4] = 0;
    raw = mcts_apply_step(&state, &cfg, plain_desc, &terminal, &credit, &plain);
    assert(raw == 6 && credit == raw && state.b2b == 7);
    state = original;
    for (int r = 36; r < 40; r++) state.board[r] = 0x3fe;
    state.board[35] = 0x200;
    int tetris_desc[] = {0, 1, 0, 36, 0};
    raw = mcts_apply_step(&state, &cfg, tetris_desc, &terminal, &credit, &plain);
    assert(raw == 5 && credit == raw && state.b2b == 7);
}

int main(void) {
    real_statistics();
    gate_boundaries();
    search_admission_and_final_choice();
    risk_backup();
    attack_credit_and_cancellation();
    return 0;
}
