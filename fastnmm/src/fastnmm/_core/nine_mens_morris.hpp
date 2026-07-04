// Fast Nine Men's Morris engine using bitboards.
// Interface designed to match OpenSpiel's nine_mens_morris game.
//
// Action encoding (matches OpenSpiel exactly):
//   0..23    : place at / capture from position p        (action = p)
//   24..599  : move from f to t                          (action = 24 + f*24 + t)
//
// Total distinct actions: 600.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace fastnmm {

// ----- Game constants -----
constexpr int kNumPositions      = 24;
constexpr int kNumPlayers        = 2;
constexpr int kPiecesPerPlayer   = 9;
constexpr int kNumDistinctActions = 600;
constexpr int kMoveActionOffset  = 24;          // moves start at action 24
constexpr int kMaxTurnsDefault   = 200;         // OpenSpiel draw threshold
constexpr int kMaxGameLength     = 1000;        // hard safety bound
constexpr int kObservationPlanes = 5;
constexpr int kObservationRows   = 7;
constexpr int kObservationCols   = 7;
constexpr int kObservationSize   = kObservationPlanes * kObservationRows * kObservationCols;

// Players (matches OpenSpiel: 0 = White, 1 = Black).
constexpr int kWhite = 0;
constexpr int kBlack = 1;

// Phase of the game.
enum class Phase : uint8_t {
    kPlacing  = 0,
    kMoving   = 1,
    kCapture  = 2,  // awaiting removal choice after a mill formed
    kTerminal = 3,
};

// 24-bit bitboard (bit i set iff position i occupied).
using BitBoard = uint32_t;
constexpr BitBoard kFullBoard = (1u << kNumPositions) - 1u;

// ----- Precomputed tables (defined in .cpp) -----
extern const BitBoard* const kAdjacency;   // length kNumPositions
extern const BitBoard* const kMillMasks;   // length 16
extern const BitBoard* const kPosMill1;    // length kNumPositions
extern const BitBoard* const kPosMill2;    // length kNumPositions
extern const int kBoardRow[kNumPositions]; // row in 7x7 display grid
extern const int kBoardCol[kNumPositions]; // col in 7x7 display grid
extern const uint8_t kObsPlane3Mask[kObservationRows][kObservationCols];
extern const uint8_t kObsPlane4Mask[kObservationRows][kObservationCols];

// ----- State -----
class State {
public:
    // Standard game: both players start with 9 stones.
    State();

    // Training variant: each player starts with `unplaced_white` /
    // `unplaced_black` stones to place. Both values must be in [1, 9].
    // Any ValueError in the input is raised as std::invalid_argument.
    State(int unplaced_white, int unplaced_black);

    // Core OpenSpiel-style API.
    std::vector<int>  LegalActions() const;
    void              ApplyAction(int action);
    bool              IsTerminal() const { return phase_ == Phase::kTerminal; }
    bool              IsChanceNode() const { return false; }
    int               CurrentPlayer() const;          // returns kTerminalPlayer if terminal
    std::array<float, 2> Returns() const;
    std::array<float, 2> Rewards() const { return last_rewards_; }
    std::vector<int>  History() const { return history_; }
    int               MoveNumber() const { return static_cast<int>(history_.size()); }
    std::string       ToString() const;
    std::string       ActionToString(int player, int action) const;
    std::string       ObservationString(int player) const { (void)player; return ToString(); }
    void              ObservationTensor(int player, float* out) const;
    std::string       Serialize() const;
    State             Clone() const { return *this; }

    // Extra helpers (useful for Python users & tests).
    BitBoard  Board(int player) const { return board_[player]; }
    int       MenToDeploy(int player) const { return unplaced_[player]; }
    int       MenOnBoard(int player) const { return on_board_[player]; }
    Phase     CurrentPhase() const;   // derives placing/moving per-player
    int       Turn() const { return turn_; }
    int       Winner() const { return winner_; }   // -1 ongoing, 0/1 player, 2 draw

    // Fast helpers that don't allocate.
    int LegalActionsInto(int* out) const;   // writes into out, returns count
    bool IsActionLegal(int action) const;

    // Static constant for "terminal" player id (matches OpenSpiel).
    static constexpr int kTerminalPlayer = -4;

    // Default draw threshold (in "turns", not actions). Configurable for advanced use.
    void SetMaxTurns(int t) { max_turns_ = t; }
    int  MaxTurns() const { return max_turns_; }

private:
    // Bitboards per player.
    BitBoard board_[2];
    int8_t   unplaced_[2];    // pieces left to place (0..9)
    int8_t   on_board_[2];    // pieces currently on the board
    int8_t   current_player_;
    Phase    phase_;
    int16_t  turn_;           // "turn" counter (like OpenSpiel: does NOT tick on captures)
    int8_t   winner_;         // -1 ongoing, 0/1 player, 2 draw
    int16_t  max_turns_;
    std::array<float, 2> last_rewards_;
    std::vector<int> history_;

    // Internal helpers.
    void ApplyPlace(int pos);
    void ApplyMove(int from, int to);
    void ApplyCapture(int pos);

    bool FormedMillAt(int pos, int player) const;
    bool PieceInMill(int pos, int player) const;
    bool HasAnyLegalMove(int player) const;

    void EndGame(int winner_player);   // 0/1 winning, or 2 for draw
    void CheckDrawAndStalemate();      // called after each "turn" transition
};

// ----- Game (static/factory). -----
struct Game {
    static int NumDistinctActions() { return kNumDistinctActions; }
    static int NumPlayers()         { return kNumPlayers; }
    static float MinUtility()       { return -1.0f; }
    static float MaxUtility()       { return 1.0f; }
    static float UtilitySum()       { return 0.0f; }
    static int MaxGameLength()      { return kMaxGameLength; }
    static std::array<int, 3> ObservationTensorShape() {
        return {kObservationPlanes, kObservationRows, kObservationCols};
    }
    static int ObservationTensorSize()   { return kObservationSize; }
    static State NewInitialState()       { return State(); }
    static State NewInitialState(int uw, int ub) { return State(uw, ub); }
    static std::string ActionToString(int player, int action);
    static State Deserialize(const std::string& data);
};

// ----- Bots -----
// Pick a random legal action (xorshift64 RNG).
int RandomAction(const State& s, uint64_t& rng);

// Depth-limited negamax with alpha-beta. Returns (best_action, score).
// Score is from the perspective of the state's current player.
// Terminal / game-over scores use kWinScore ± depth tie-breaking.
// Higher `depth` => stronger but slower.
struct SearchResult {
    int action;
    int score;
    long nodes_visited;
};
SearchResult MinimaxSearch(const State& s, int depth);

// Static evaluation heuristic (from perspective of `s.CurrentPlayer()`).
int Evaluate(const State& s);

// Detailed breakdown of the evaluation, useful for debugging dashboards.
// All counts are *own minus opponent* differentials from the perspective
// of `s.CurrentPlayer()`. Weighted contributions sum to `total` (which
// matches `Evaluate(s)` for non-terminal states).
struct EvalBreakdown {
    int own_material;
    int opp_material;
    int own_mills;
    int opp_mills;
    int own_open_mills;
    int opp_open_mills;
    int own_running_mills;
    int opp_running_mills;
    int own_mill_blocks;
    int opp_mill_blocks;
    int own_blocked;
    int opp_blocked;
    int own_mobility;
    int opp_mobility;
    int w_material;
    int w_mill;
    int w_open_mill;
    int w_running_mill;
    int w_double_mill;
    int w_mill_block;
    int w_blocked;
    int w_mobility;
    int material_score;
    int mill_score;
    int open_mill_score;
    int running_mill_score;
    int double_mill_score;
    int mill_block_score;
    int blocked_score;
    int mobility_score;
    int total;
    bool endgame;
    int current_player;
};
EvalBreakdown EvaluateBreakdown(const State& s);

// Tunable evaluation weights. Defaults reproduce the hardcoded weights
// used by Evaluate(const State&). Per-game jitter (option 1 in the
// non-determinism plan) lets callers vary these to make the bot pick
// different tactics across games without changing the search core.
struct EvalWeights {
    int w_material_mid = 8;
    int w_material_end = 14;
    int w_mill_mid     = 18;
    int w_mill_end     = 22;
    int w_open_mid     = 14;
    int w_open_end     = 22;
    int w_running      = 30;
    int w_double       = 28;
    int w_mill_block   = 10;
    int w_blocked_mid  = 10;   // endgame blocked weight is fixed at 0
    int w_mobility_mid = 1;    // endgame mobility weight is fixed at 0
};

// Weighted variant of Evaluate. The unweighted overload above stays
// compile-time-constant for the hot free-function MinimaxSearch path.
int Evaluate(const State& s, const EvalWeights& w);

// =========================================================================
// AI / reward-shaping evaluator.
//
// `Evaluate(s)` above is tuned for alpha-beta search: int weights chosen
// for pruning quality, evaluated from `s.CurrentPlayer()`'s perspective.
// PPO reward shaping has different needs:
//   - it always evaluates from the *acting* player's view (which may not
//     be the side-to-move after `apply_action` returns), so we take an
//     explicit `player` argument;
//   - it wants raw feature counts that Python can combine with the
//     curriculum's float weights (mill_reward, mobility_reward, ...)
//     and scale by `shaping_multiplier`. Returning a single int here
//     would force the weights to live in C++.
//
// Both functions are pure C++ that release the GIL when bound -- the
// hot-path call replaces a parse-board + 7 Python loops + dict build per
// AI move, which is most of `worker.py`'s per-move overhead.
// =========================================================================
struct AIEvalBreakdown {
    // Raw counts from the perspective of the `player` argument passed in.
    // Field names mirror the dict keys produced by
    // `utils.extract_state_features` so the Python migration is mechanical.
    int my_pieces;
    int opp_pieces;
    int my_mills;
    int opp_mills;
    int my_potential_mills;        // == own open mills (closable next turn)
    int opp_potential_mills;
    int my_blocked_mills;          // own stone in a 2-opp/1-own mill line
    int opp_blocked_mills;
    int my_unblocked_threats;      // opp's potential_mills (== opp_open)
    int opp_unblocked_threats;
    int my_double_mills;           // running/swing-mill setups (own)
    int opp_double_mills;
    int my_mobility;
    int opp_mobility;
    int player;                    // echoes the input `player` for safety
    bool endgame;                  // either side is flying
};
AIEvalBreakdown EvaluateForAIBreakdown(const State& s, int player);
int              EvaluateForAI(const State& s, int player);

// =========================================================================
// MinimaxEngine: persistent transposition-table-backed alpha-beta.
//
// Bit-exact parity with MinimaxSearch (same returned score, same chosen
// root move for any state/depth) -- the TT only accelerates the search,
// it does not change move selection. A test in tests/ verifies parity
// across many random states.
//
// Construction:
//   MinimaxEngine eng(tt_bytes = 256 MiB);
//
// Memory:
//   tt_bytes is rounded down to the largest power-of-two number of
//   16-byte entries. With 180 GiB of RAM you can configure
//   MinimaxEngine(128ULL << 30) for ~8.6 G entries.
//
// Lifecycle:
//   - The TT persists across Search() calls. Call NewGame() between
//     unrelated games to bump the generation tag (used as a tiebreak in
//     the replacement policy so older entries get evicted preferentially).
//   - TtClear() wipes the table.
//
// Statistics:
//   probes / hits / stores are exposed for monitoring; they are reset
//   only by TtClear().
// =========================================================================
class MinimaxEngine {
public:
    // strict_parity:
    //   true (default)  -- bit-exact match with MinimaxSearch. TT
    //                      probes require cached_depth == requested_depth.
    //   false           -- position-relative mate-distance scoring +
    //                      cached_depth >= requested_depth probe. Note:
    //                      in normal forward minimax this gives the same
    //                      hit rate as strict (because all paths to a
    //                      node reach it at the same remaining depth);
    //                      relaxed only outperforms strict in patterns
    //                      where shallow queries follow deeper stores
    //                      (e.g. opening books). Kept as an opt-in.
    explicit MinimaxEngine(std::size_t tt_bytes = 256ULL * 1024 * 1024,
                           bool strict_parity = true);
    ~MinimaxEngine();

    MinimaxEngine(const MinimaxEngine&)            = delete;
    MinimaxEngine& operator=(const MinimaxEngine&) = delete;

    SearchResult Search(const State& s, int depth);
    int          Eval(const State& s) const;
    bool         StrictParity() const { return strict_; }

    // Per-engine evaluation weights. Updating clears the TT because
    // stored scores were computed under the old weights and would
    // poison the new search.
    const EvalWeights& Weights() const { return weights_; }
    void               SetWeights(const EvalWeights& w);

    // Bump generation -- old entries become eligible for replacement.
    void NewGame();
    // Wipe the TT, reset stats.
    void TtClear();

    // Stats (zeroed only by TtClear).
    std::size_t TtNumEntries() const { return num_entries_; }
    std::size_t TtBytes()      const;
    std::size_t TtProbes()     const { return probes_; }
    std::size_t TtHits()       const { return hits_; }
    std::size_t TtStores()     const { return stores_; }
    std::size_t TtCollisions() const { return collisions_; }
    // Approximate fraction of slots currently filled with a current-gen entry.
    double      TtFillFraction() const;

private:
    struct alignas(16) TTEntry {
        uint64_t key;        // Zobrist key
        int32_t  score;      // negamax score from the position's POV
        int16_t  best_move;  // -1 if none
        int8_t   depth;      // ply remaining when stored
        uint8_t  flag_age;   // bits 0..1 = bound flag, bits 2..7 = generation
    };
    static_assert(sizeof(TTEntry) == 16, "TTEntry must be 16 bytes");

    enum Flag : uint8_t { kFlagExact = 0, kFlagLower = 1, kFlagUpper = 2 };

    // Two recursion paths -- bit-exact strict, or relaxed-mate-distance.
    int NegamaxStrict(const State& s, int depth, int alpha, int beta,
                      long& nodes, uint64_t key);
    int NegamaxRelaxed(const State& s, int depth, int alpha, int beta,
                       long& nodes, uint64_t key);

    // Helpers.
    void  Allocate(std::size_t tt_bytes);
    bool  ProbeStrict(uint64_t key, int depth, int alpha, int beta,
                      int& out_score, int& out_move) const;
    bool  ProbeRelaxed(uint64_t key, int depth, int alpha, int beta,
                       int& out_score, int& out_move) const;
    void  Store(uint64_t key, int depth, int score, int move, Flag flag);
    inline std::size_t Index(uint64_t key) const { return key & mask_; }

    TTEntry*         table_       = nullptr;
    std::size_t      num_entries_ = 0;     // power of two
    uint64_t         mask_        = 0;
    uint8_t          generation_  = 0;
    bool             strict_      = false;
    bool             use_custom_weights_ = false;
    EvalWeights      weights_     = {};

    mutable std::size_t probes_      = 0;
    mutable std::size_t hits_        = 0;
    std::size_t         stores_      = 0;
    std::size_t         collisions_  = 0;
};

// =========================================================================
// Opponent stepping: play_until_player.
//
// Helper used to keep the Python AI training loop in C++ for the
// opponent's turns. Given a state and a `target_player`, repeatedly
// applies opponent actions in C++ until the state is terminal or it
// becomes `target_player`'s move.
//
// Notes:
//   - The state is mutated in place. The list of actions taken is
//     returned (useful for logging / replay).
//   - For minimax opponents, an existing `engine` is reused so its TT
//     persists; pass nullptr for random opponents.
//   - random_move_prob lets you mix exploration into a minimax opponent
//     (matching the existing _RandomizedMinimaxBot wrapper). Set to
//     0 for pure minimax, 1 for fully random behaviour.
//   - A per-call rng_state is threaded through xorshift64 so callers
//     get reproducibility when they pass a fixed seed.
// =========================================================================
enum class OpponentKind : uint8_t {
    kRandom  = 0,
    kMinimax = 1,
};

struct OpponentSpec {
    OpponentKind kind             = OpponentKind::kRandom;
    int          minimax_depth    = 4;
    double       random_move_prob = 0.0;
};

std::vector<int> PlayUntilPlayer(State& state,
                                 int target_player,
                                 const OpponentSpec& opp,
                                 MinimaxEngine* engine,
                                 uint64_t& rng_state);

// Compute the Zobrist key of a state. Exposed for testing / external
// caches; the engine maintains its own (recomputes on probe).
uint64_t ZobristKey(const State& s);

// Inline decoding helpers.
inline bool IsMoveAction(int action)  { return action >= kMoveActionOffset; }
inline int  MoveFrom(int action)      { return (action - kMoveActionOffset) / kNumPositions; }
inline int  MoveTo(int action)        { return (action - kMoveActionOffset) % kNumPositions; }
inline int  EncodeMove(int from, int to) {
    return kMoveActionOffset + from * kNumPositions + to;
}

// Run N random rollouts from the initial state, returning the sum of player-0
// returns (useful for pure-C++ benchmarking and Monte Carlo estimates).
// When `out_lengths` is non-null, writes the length of each game (in actions).
double RandomPlayouts(int num_games, uint64_t seed,
                      int* out_lengths = nullptr);

// =========================================================================
// ParallelMinimaxBot: root-splitting parallel alpha-beta.
//
// Holds `num_threads` persistent MinimaxEngine instances plus a persistent
// worker pool. On each Search call the root's legal actions are partitioned
// round-robin across workers; each worker scores its slice and a global
// argmax reduces to (best_action, best_score, total_nodes).
//
// Why root-split and not a shared TT search? Per-thread TTs avoid the
// concurrent-hashmap synchronisation cost and match the engine's existing
// data-flow exactly -- each engine builds its own TT as the game
// progresses. The trade-off is no cross-thread transposition sharing and
// no narrowed root alpha (we use a full window per worker). For depths
// 4-10 in Nine Men's Morris this gives ~1.5-3x speedup with 4-6 workers.
//
// Bit-exact parity with single-threaded MinimaxEngine::Search for terminal
// leaves (kWinScore +/- depth tie-break). Non-terminal leaves use the
// engine's strict_parity mode (same TT/probe behaviour).
//
// Thread-safety: a single ParallelMinimaxBot instance is NOT safe to call
// Search() on from multiple threads concurrently. Construct one per game.
// =========================================================================
class ParallelMinimaxBot {
public:
    ParallelMinimaxBot(int num_threads,
                       std::size_t tt_bytes_per_thread,
                       bool strict_parity = true);
    ~ParallelMinimaxBot();

    ParallelMinimaxBot(const ParallelMinimaxBot&)            = delete;
    ParallelMinimaxBot& operator=(const ParallelMinimaxBot&) = delete;

    SearchResult Search(const State& s, int depth);

    int NumThreads() const { return num_threads_; }

private:
    struct Impl;
    Impl* impl_;
    int   num_threads_;
};

// =========================================================================
// Progressive minimax evaluation.
//
// Runs the full model-vs-minimax climb in pure C++:
//
//   for depth in 1, 2, 3, ... (until win_rate <= 0.5 or max_depth):
//       run `games_per_depth` games concurrently
//           each game uses its own ParallelMinimaxBot with
//           `threads_per_game` workers
//       on minimax turns: pure-C++ alpha-beta (GIL released)
//       on model turns: re-acquire GIL, call `policy_fn(state) -> action`
//
// total threads in flight at peak = games_per_depth * threads_per_game.
// For a 24-core machine, e.g. 6 games x 4 threads/game saturates all cores
// during the bot search (which is where >95% of wall-time is spent).
//
// PolicyFn signature (called per model decision, with GIL re-acquired):
//   int policy_fn(const State& state, int current_player,
//                 int game_idx, int depth) -> action_int
//
// Returns a map: depth -> { wins, draws, losses, total_nodes, win_rate, ... }.
// max_depth_beaten is the largest depth where win_rate > 0.5.
// =========================================================================
struct GameSpec {
    int unplaced_white = 9;
    int unplaced_black = 9;
    int ai_player      = 0;  // 0 = AI is white, 1 = AI is black
};

struct DepthResult {
    int  wins        = 0;
    int  draws       = 0;
    int  losses      = 0;
    long total_nodes = 0;
    double win_rate  = 0.0;
    double wall_seconds = 0.0;
};

struct ProgressiveEvalConfig {
    int    max_depth          = 100;          // upper bound when unlimited=false
    int    games_per_depth    = 6;
    int    max_threads        = 24;           // total threads in flight at peak
    int    max_steps          = 200;          // per-game step cap (draw on cap)
    std::size_t tt_bytes_per_thread = 64ULL * 1024 * 1024;
    bool   unlimited          = true;         // climb until win_rate <= 0.5
    bool   strict_parity      = true;
    double time_budget_s      = -1.0;         // negative = no budget
};

// PolicyFn: called whenever it's the AI's turn. Must return a legal action.
// The state is read-only from the callback's perspective; the caller will
// apply the returned action after the callback returns.
using ProgressivePolicyFn = std::function<int(const State&, int /*current_player*/,
                                              int /*game_idx*/, int /*depth*/)>;

// Per-depth progress callback. Optional. Fired after each depth completes
// (after the climb decision is made). Useful for streaming D{n}:W/D/L lines
// to a logger as soon as a depth finishes.
using ProgressiveProgressFn = std::function<void(int /*depth*/, const DepthResult&)>;

// GameSpec provider: called once per game, gives caller control over
// stone counts and which side the AI plays. Default behaviour (when
// nullptr): ai_player alternates by game index, both sides start with 9.
using ProgressiveGameSpecFn = std::function<GameSpec(int /*depth*/, int /*game_idx*/)>;

struct ProgressiveEvalResult {
    int                              max_depth_beaten = 0;
    std::map<int, DepthResult>       per_depth;
    double                           total_wall_seconds = 0.0;
};

ProgressiveEvalResult RunProgressiveEval(const ProgressiveEvalConfig& cfg,
                                         const ProgressivePolicyFn& policy_fn,
                                         const ProgressiveGameSpecFn& spec_fn = nullptr,
                                         const ProgressiveProgressFn& progress_fn = nullptr);

}  // namespace fastnmm
