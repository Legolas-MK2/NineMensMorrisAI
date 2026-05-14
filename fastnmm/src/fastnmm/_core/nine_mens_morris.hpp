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
#include <atomic>
#include <cstddef>
#include <cstdint>
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

class SharedMoveCache;  // forward decl; full definition below.

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

    // Optional shared-memory best-move cache, consulted at the search
    // root for move-ordering and updated with the chosen root action
    // after the search completes. Engine does NOT take ownership; the
    // pointer must outlive the engine. Pass nullptr to detach.
    void SetRootCache(SharedMoveCache* cache) { root_cache_ = cache; }
    SharedMoveCache* RootCache() const { return root_cache_; }

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
    SharedMoveCache* root_cache_  = nullptr;  // non-owning

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

// =========================================================================
// SharedMoveCache: lock-free, cross-process best-move cache.
//
// Maps Zobrist position key -> last best-move chosen at that position by
// any worker that ever searched it. Backed by POSIX shared memory so all
// 22 training workers can read/write a single table.
//
// Layout: open-addressing linear probing, 16-byte aligned entries:
//   { atomic<uint64_t> key; atomic<int32_t> action; uint32_t pad; }
// Empty slot = key == 0. (Zobrist key 0 is rejected to keep this sentinel
// unambiguous; collision probability is 2^-64, negligible.)
//
// Concurrency: the action-before-key write order means a torn reader who
// observes a fresh action with a stale key gets a key mismatch and falls
// through. Wrong cached actions are validated against legal_actions() by
// the consumer (engine), so even a benign race can't corrupt search.
//
// Sizing: 16 B/entry. 1 GiB -> ~64 M entries; 16 GiB -> ~1 G entries.
// =========================================================================
class SharedMoveCache {
public:
    static constexpr int kNoMove   = -1;
    static constexpr int kProbeMax = 16;
    static constexpr std::size_t kBytesPerEntry = 16;

    // Create or attach. `name` must start with '/' (POSIX shm_open
    // convention). `total_bytes` is honoured only on create; attachers
    // discover the actual size from the existing segment.
    SharedMoveCache(const std::string& name,
                    std::size_t total_bytes,
                    bool create);
    ~SharedMoveCache();

    SharedMoveCache(const SharedMoveCache&)            = delete;
    SharedMoveCache& operator=(const SharedMoveCache&) = delete;

    // Returns the cached action, or kNoMove on miss.
    int  Get(uint64_t key) const;
    // Stores the action at `key`. Silently no-op if the probe limit is
    // exhausted (table is locally dense at that bucket).
    void Put(uint64_t key, int action);

    // Stats are per-process (counters in this object, not in shm).
    std::size_t NumEntries() const { return num_entries_; }
    std::size_t Bytes()      const { return bytes_; }
    std::size_t Probes()     const { return probes_; }
    std::size_t Hits()       const { return hits_; }
    std::size_t Stores()     const { return stores_; }
    std::size_t StoreMisses() const { return store_misses_; }
    bool        IsCreator()  const { return is_creator_; }
    const std::string& Name() const { return name_; }

    // Manual lifecycle. Destructor calls Close() automatically; only the
    // creator should call Unlink() to remove the segment from the system.
    void Close();
    void Unlink();

private:
    struct alignas(16) Entry {
        std::atomic<uint64_t> key;
        std::atomic<int32_t>  action;
        uint32_t              pad;
    };
    static_assert(sizeof(Entry) == 16, "Entry must be 16 bytes");

    Entry*      table_       = nullptr;
    std::size_t num_entries_ = 0;     // power of two
    uint64_t    mask_        = 0;
    std::size_t bytes_       = 0;
    int         shm_fd_      = -1;
    std::string name_;
    bool        is_creator_  = false;
    bool        closed_      = false;

    mutable std::size_t probes_       = 0;
    mutable std::size_t hits_         = 0;
    std::size_t         stores_       = 0;
    std::size_t         store_misses_ = 0;
};

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

}  // namespace fastnmm
