// Fast Nine Men's Morris engine -- implementation.

#include "nine_mens_morris.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace fastnmm {

// =========================================================================
// Precomputed tables.
// =========================================================================

namespace {

// Adjacency list: adjacency[i] = list of positions adjacent to i.
constexpr int kAdj[kNumPositions][4] = {
    {  1,  9, -1, -1},   //  0
    {  0,  2,  4, -1},   //  1
    {  1, 14, -1, -1},   //  2
    {  4, 10, -1, -1},   //  3
    {  1,  3,  5,  7},   //  4
    {  4, 13, -1, -1},   //  5
    {  7, 11, -1, -1},   //  6
    {  4,  6,  8, -1},   //  7
    {  7, 12, -1, -1},   //  8
    {  0, 10, 21, -1},   //  9
    {  3,  9, 11, 18},   // 10
    {  6, 10, 15, -1},   // 11
    {  8, 13, 17, -1},   // 12
    {  5, 12, 14, 20},   // 13
    {  2, 13, 23, -1},   // 14
    { 11, 16, -1, -1},   // 15
    { 15, 17, 19, -1},   // 16
    { 12, 16, -1, -1},   // 17
    { 10, 19, -1, -1},   // 18
    { 16, 18, 20, 22},   // 19
    { 13, 19, -1, -1},   // 20
    {  9, 22, -1, -1},   // 21
    { 19, 21, 23, -1},   // 22
    { 14, 22, -1, -1},   // 23
};

// 16 mills (3-in-a-row lines).
constexpr int kMills[16][3] = {
    // Horizontal mills:
    { 0,  1,  2}, { 3,  4,  5}, { 6,  7,  8},
    { 9, 10, 11}, {12, 13, 14},
    {15, 16, 17}, {18, 19, 20}, {21, 22, 23},
    // Vertical mills:
    { 0,  9, 21}, { 3, 10, 18}, { 6, 11, 15},
    { 1,  4,  7}, {16, 19, 22},
    { 8, 12, 17}, { 5, 13, 20}, { 2, 14, 23},
};

// 7x7 display coordinates for each position (row, col).
constexpr int kCoords[kNumPositions][2] = {
    {0,0}, {0,3}, {0,6},
    {1,1}, {1,3}, {1,5},
    {2,2}, {2,3}, {2,4},
    {3,0}, {3,1}, {3,2},
    {3,4}, {3,5}, {3,6},
    {4,2}, {4,3}, {4,4},
    {5,1}, {5,3}, {5,5},
    {6,0}, {6,3}, {6,6},
};

// Helper to build bitboard from a list of positions.
constexpr BitBoard MakeMask(const int* pts, int n) {
    BitBoard m = 0;
    for (int i = 0; i < n; ++i) m |= (1u << pts[i]);
    return m;
}

}  // anonymous namespace

// Initialize tables at program start.
namespace detail {

std::array<BitBoard, kNumPositions> InitAdjacency() {
    std::array<BitBoard, kNumPositions> arr{};
    for (int i = 0; i < kNumPositions; ++i) {
        BitBoard m = 0;
        for (int j = 0; j < 4; ++j) {
            if (kAdj[i][j] >= 0) m |= (1u << kAdj[i][j]);
        }
        arr[i] = m;
    }
    return arr;
}

std::array<BitBoard, 16> InitMillMasks() {
    std::array<BitBoard, 16> arr{};
    for (int i = 0; i < 16; ++i) {
        arr[i] = (1u << kMills[i][0]) | (1u << kMills[i][1]) | (1u << kMills[i][2]);
    }
    return arr;
}

// For each position, the two mill masks it belongs to. Every position
// participates in exactly 2 mills (one horizontal, one vertical).
void InitPositionMills(std::array<BitBoard, kNumPositions>& p1,
                       std::array<BitBoard, kNumPositions>& p2) {
    p1.fill(0);
    p2.fill(0);
    for (int pos = 0; pos < kNumPositions; ++pos) {
        BitBoard first = 0;
        for (int m = 0; m < 16; ++m) {
            BitBoard mask = (1u << kMills[m][0]) | (1u << kMills[m][1]) | (1u << kMills[m][2]);
            if (mask & (1u << pos)) {
                if (first == 0) { p1[pos] = mask; first = mask; }
                else            { p2[pos] = mask; break; }
            }
        }
    }
}

}  // namespace detail

// Static init via function-local statics (thread-safe in C++11).
namespace {
const auto& Adjacency() {
    static const auto v = detail::InitAdjacency();
    return v;
}
const auto& MillMasksArr() {
    static const auto v = detail::InitMillMasks();
    return v;
}
struct PosMills { std::array<BitBoard, kNumPositions> p1, p2; };
const PosMills& PositionMills() {
    static const PosMills v = [] {
        PosMills pm;
        detail::InitPositionMills(pm.p1, pm.p2);
        return pm;
    }();
    return v;
}
}  // namespace

// Public-accessible arrays (exposed via extern declarations).
const BitBoard* const kAdjacency = Adjacency().data();
const BitBoard* const kMillMasks = MillMasksArr().data();
const BitBoard* const kPosMill1  = PositionMills().p1.data();
const BitBoard* const kPosMill2  = PositionMills().p2.data();
const int       kBoardRow[kNumPositions] = {
    0,0,0, 1,1,1, 2,2,2, 3,3,3, 3,3,3, 4,4,4, 5,5,5, 6,6,6
};
const int       kBoardCol[kNumPositions] = {
    0,3,6, 1,3,5, 2,3,4, 0,1,2, 4,5,6, 2,3,4, 1,3,5, 0,3,6
};

// Structural observation-tensor planes (3 and 4). These are constant masks
// that encode the board's geometric edges, matching OpenSpiel's output.
// Derived empirically from the OpenSpiel reference engine.
const uint8_t kObsPlane3Mask[kObservationRows][kObservationCols] = {
    {0,1,1,0,1,1,0},
    {0,0,1,0,1,0,0},
    {0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0},
    {0,0,1,0,1,0,0},
    {0,1,1,0,1,1,0},
};
const uint8_t kObsPlane4Mask[kObservationRows][kObservationCols] = {
    {0,0,0,0,0,0,0},
    {1,0,0,0,0,0,1},
    {1,1,0,0,0,1,1},
    {0,0,0,0,0,0,0},
    {1,1,0,0,0,1,1},
    {1,0,0,0,0,0,1},
    {0,0,0,0,0,0,0},
};

// =========================================================================
// State
// =========================================================================

State::State() : State(kPiecesPerPlayer, kPiecesPerPlayer) {}

State::State(int unplaced_white, int unplaced_black)
    : board_{0, 0},
      unplaced_{static_cast<int8_t>(unplaced_white),
                static_cast<int8_t>(unplaced_black)},
      on_board_{0, 0},
      current_player_(kWhite),
      phase_(Phase::kPlacing),
      turn_(0),
      winner_(-1),
      max_turns_(kMaxTurnsDefault),
      last_rewards_{0.0f, 0.0f},
      history_{} {
    if (unplaced_white < 1 || unplaced_white > kPiecesPerPlayer ||
        unplaced_black < 1 || unplaced_black > kPiecesPerPlayer) {
        throw std::invalid_argument(
            "starting stones must be in [1, 9] for each player");
    }
    history_.reserve(256);
}

int State::CurrentPlayer() const {
    return (phase_ == Phase::kTerminal) ? kTerminalPlayer : current_player_;
}

std::array<float, 2> State::Returns() const {
    if (phase_ != Phase::kTerminal) return {0.0f, 0.0f};
    if (winner_ == 2)               return {0.0f, 0.0f};   // draw
    if (winner_ == 0)               return {1.0f, -1.0f};
    /* winner_ == 1 */              return {-1.0f, 1.0f};
}

Phase State::CurrentPhase() const {
    if (phase_ == Phase::kTerminal) return Phase::kTerminal;
    if (phase_ == Phase::kCapture)  return Phase::kCapture;
    return (unplaced_[current_player_] > 0) ? Phase::kPlacing : Phase::kMoving;
}

// ----- Mill detection.
bool State::FormedMillAt(int pos, int player) const {
    const BitBoard b = board_[player];
    const BitBoard m1 = kPosMill1[pos];
    const BitBoard m2 = kPosMill2[pos];
    return ((b & m1) == m1) || (m2 != 0 && (b & m2) == m2);
}

bool State::PieceInMill(int pos, int player) const {
    return FormedMillAt(pos, player);
}

// ----- Legal-move computation.
bool State::HasAnyLegalMove(int player) const {
    const BitBoard own = board_[player];
    const BitBoard occ = board_[0] | board_[1];
    const BitBoard empty = (~occ) & kFullBoard;

    // Flying? (exactly 3 pieces, 0 unplaced.)
    const bool flying = (on_board_[player] == 3) && (unplaced_[player] == 0);
    if (flying) {
        // Any own piece and any empty square ==> always at least one move
        // (since own >= 3 and empty >= 1).
        return (own != 0) && (empty != 0);
    }
    // Standard movement: any own piece with an empty adjacent square.
    BitBoard pieces = own;
    while (pieces) {
        const int pos = __builtin_ctz(pieces);
        pieces &= pieces - 1;
        if (kAdjacency[pos] & empty) return true;
    }
    return false;
}

int State::LegalActionsInto(int* out) const {
    if (phase_ == Phase::kTerminal) return 0;

    int n = 0;

    if (phase_ == Phase::kCapture) {
        // Remove opponent piece. Prefer non-mill pieces; if all opponent pieces
        // are in mills, any of them can be captured.
        const int opp = 1 - current_player_;
        const BitBoard opp_b = board_[opp];
        BitBoard non_mill = opp_b;
        const auto& millsA = MillMasksArr();
        for (int i = 0; i < 16; ++i) {
            if ((opp_b & millsA[i]) == millsA[i]) {
                non_mill &= ~millsA[i];
            }
        }
        BitBoard choices = (non_mill != 0) ? non_mill : opp_b;
        while (choices) {
            const int p = __builtin_ctz(choices);
            choices &= choices - 1;
            out[n++] = p;
        }
        return n;
    }

    // Placing vs. moving is per-player: if this player still has stones
    // to deploy, they place; otherwise they move (possibly flying).
    if (unplaced_[current_player_] > 0) {
        const BitBoard empty = (~(board_[0] | board_[1])) & kFullBoard;
        BitBoard m = empty;
        while (m) {
            const int p = __builtin_ctz(m);
            m &= m - 1;
            out[n++] = p;
        }
        return n;
    }

    // Moving phase for this player.
    const BitBoard own = board_[current_player_];
    const BitBoard occ = board_[0] | board_[1];
    const BitBoard empty = (~occ) & kFullBoard;
    const bool flying = (on_board_[current_player_] == 3) && (unplaced_[current_player_] == 0);

    BitBoard pieces = own;
    while (pieces) {
        const int from = __builtin_ctz(pieces);
        pieces &= pieces - 1;
        BitBoard targets = flying ? empty : (kAdjacency[from] & empty);
        while (targets) {
            const int to = __builtin_ctz(targets);
            targets &= targets - 1;
            out[n++] = EncodeMove(from, to);
        }
    }
    return n;
}

std::vector<int> State::LegalActions() const {
    if (phase_ == Phase::kTerminal) return {};
    // Upper bound: moving phase flying worst case = 9 pieces * 21 empties = 189.
    int buf[256];
    const int n = LegalActionsInto(buf);
    return std::vector<int>(buf, buf + n);
}

bool State::IsActionLegal(int action) const {
    int buf[256];
    const int n = LegalActionsInto(buf);
    for (int i = 0; i < n; ++i) if (buf[i] == action) return true;
    return false;
}

// ----- Apply a single action.
void State::ApplyAction(int action) {
    if (phase_ == Phase::kTerminal) {
        throw std::runtime_error("ApplyAction called on terminal state.");
    }
    last_rewards_ = {0.0f, 0.0f};

    if (phase_ == Phase::kCapture) {
        if (action < 0 || action >= kNumPositions) {
            throw std::runtime_error("Invalid capture action.");
        }
        ApplyCapture(action);
    } else if (unplaced_[current_player_] > 0) {
        // This player still has stones to deploy -> placing.
        if (action < 0 || action >= kNumPositions) {
            throw std::runtime_error("Invalid place action.");
        }
        ApplyPlace(action);
    } else {
        // This player is in moving phase (even if the other still places).
        if (!IsMoveAction(action) || action >= kNumDistinctActions) {
            throw std::runtime_error("Invalid move action in moving phase.");
        }
        ApplyMove(MoveFrom(action), MoveTo(action));
    }

    history_.push_back(action);
}

void State::ApplyPlace(int pos) {
    const int p = current_player_;
    if (((board_[0] | board_[1]) >> pos) & 1) {
        throw std::runtime_error("Cannot place on occupied point.");
    }
    if (unplaced_[p] <= 0) {
        throw std::runtime_error("No pieces left to place.");
    }
    board_[p]    |= (1u << pos);
    unplaced_[p] -= 1;
    on_board_[p] += 1;

    const bool mill = FormedMillAt(pos, p);
    if (mill) {
        // Enter capture phase (current player must pick opponent piece).
        phase_ = Phase::kCapture;
        return;
    }

    // Advance: next player.
    turn_ += 1;
    current_player_ = 1 - p;

    CheckDrawAndStalemate();
}

void State::ApplyMove(int from, int to) {
    const int p = current_player_;
    if (!((board_[p] >> from) & 1)) {
        throw std::runtime_error("Source position does not contain own piece.");
    }
    if (((board_[0] | board_[1]) >> to) & 1) {
        throw std::runtime_error("Destination is occupied.");
    }
    const bool flying = (on_board_[p] == 3) && (unplaced_[p] == 0);
    if (!flying && !(kAdjacency[from] & (1u << to))) {
        throw std::runtime_error("Non-adjacent move while not in flying phase.");
    }
    board_[p] ^= (1u << from);
    board_[p] |= (1u << to);

    if (FormedMillAt(to, p)) {
        phase_ = Phase::kCapture;
        return;
    }

    turn_ += 1;
    current_player_ = 1 - p;
    CheckDrawAndStalemate();
}

void State::ApplyCapture(int pos) {
    const int p = current_player_;
    const int opp = 1 - p;
    if (!((board_[opp] >> pos) & 1)) {
        throw std::runtime_error("Capture target not an opponent piece.");
    }
    // Enforce "cannot take from mill unless all in mills" rule.
    const BitBoard opp_b = board_[opp];
    BitBoard non_mill = opp_b;
    for (int i = 0; i < 16; ++i) {
        if ((opp_b & kMillMasks[i]) == kMillMasks[i]) {
            non_mill &= ~kMillMasks[i];
        }
    }
    if (non_mill != 0 && !((non_mill >> pos) & 1)) {
        throw std::runtime_error(
            "Cannot capture piece in a mill while opponent has non-mill pieces.");
    }

    board_[opp]    ^= (1u << pos);
    on_board_[opp] -= 1;

    // After capture, control returns to the opponent (per standard rules).
    turn_ += 1;
    current_player_ = opp;
    phase_ = Phase::kPlacing;   // nominal; effective phase is derived in
                                // LegalActions from unplaced_[current_player_]

    // Check win by piece count: the captured side loses if they have
    // no more stones to deploy AND < 3 stones on the board.
    if (unplaced_[opp] == 0 && on_board_[opp] < 3) {
        EndGame(p);
        return;
    }
    CheckDrawAndStalemate();
}

void State::EndGame(int winner_player) {
    // winner_player: 0 or 1 for a winning player, 2 for draw.
    phase_ = Phase::kTerminal;
    winner_ = static_cast<int8_t>(winner_player);
    if (winner_player == 2) {
        last_rewards_ = {0.0f, 0.0f};
    } else if (winner_player == 0) {
        last_rewards_ = {1.0f, -1.0f};
    } else {
        last_rewards_ = {-1.0f, 1.0f};
    }
}

void State::CheckDrawAndStalemate() {
    // Max-turn draw applies in all non-terminal phases.
    if (turn_ >= max_turns_) { EndGame(2); return; }

    // Stalemate applies only when the current player is in their *moving*
    // sub-phase (they have no stones left to place). A player who still has
    // stones to place can always play (empty points are plentiful).
    if (phase_ == Phase::kCapture) return;
    if (unplaced_[current_player_] > 0) return;   // still placing
    if (!HasAnyLegalMove(current_player_)) {
        EndGame(1 - current_player_);
    }
}

// ----- String rendering.
// Render in OpenSpiel's ASCII format, using W/B/'.' for white/black/empty.
std::string State::ToString() const {
    // The OpenSpiel layout:
    //   .------.------.
    //   |      |      |
    //   | .----.----. |
    //   | |    |    | |
    //   | | .--.--. | |
    //   | | |     | | |
    //   .-.-.     .-.-.
    //   | | |     | | |
    //   | | .--.--. | |
    //   | |    |    | |
    //   | .----.----. |
    //   |      |      |
    //   .------.------.
    //
    // The positions 0..23 show up at specific character columns.
    const char glyph[3] = {'W', 'B', '.'};

    auto at = [&](int idx) -> char {
        if ((board_[0] >> idx) & 1) return glyph[0];
        if ((board_[1] >> idx) & 1) return glyph[1];
        return glyph[2];
    };

    std::ostringstream os;
    os << at( 0) << "------" << at( 1) << "------" << at( 2) << "\n";
    os << "|      |      |\n";
    os << "| "    << at( 3) << "----"   << at( 4) << "----"   << at( 5) << " |\n";
    os << "| |    |    | |\n";
    os << "| | "  << at( 6) << "--"     << at( 7) << "--"     << at( 8) << " | |\n";
    os << "| | |     | | |\n";
    os << at( 9) << "-"     << at(10)   << "-"     << at(11)
       << "     "
       << at(12) << "-"     << at(13)   << "-"     << at(14) << "\n";
    os << "| | |     | | |\n";
    os << "| | "  << at(15) << "--"     << at(16) << "--"     << at(17) << " | |\n";
    os << "| |    |    | |\n";
    os << "| "    << at(18) << "----"   << at(19) << "----"   << at(20) << " |\n";
    os << "|      |      |\n";
    os << at(21) << "------" << at(22) << "------" << at(23) << "\n";
    os << "\n";
    os << "Current player: "
       << (phase_ == Phase::kTerminal
           ? (winner_ == 2 ? "-" : (winner_ == 0 ? "W" : "B"))
           : (current_player_ == 0 ? "W" : "B"))
       << "\n";
    os << "Turn number: " << turn_ << "\n";
    os << "Men to deploy: " << static_cast<int>(unplaced_[0]) << " "
                           << static_cast<int>(unplaced_[1]) << "\n";
    os << "Num men: "
       << static_cast<int>(on_board_[0] + unplaced_[0]) << " "
       << static_cast<int>(on_board_[1] + unplaced_[1]) << "\n";
    if (phase_ == Phase::kCapture) {
        os << "Last move formed a mill. Capture time!\n";
    }
    return os.str();
}

std::string State::ActionToString(int player, int action) const {
    (void)player;
    return Game::ActionToString(player, action);
}

std::string Game::ActionToString(int /*player*/, int action) {
    if (action < 0 || action >= kNumDistinctActions) {
        return "InvalidAction";
    }
    if (IsMoveAction(action)) {
        std::ostringstream os;
        os << "Move " << MoveFrom(action) << " -> " << MoveTo(action);
        return os.str();
    }
    std::ostringstream os;
    os << "Point " << action;
    return os.str();
}

// ----- Observation tensor (5,7,7).
// Plane 0: White pieces
// Plane 1: Black pieces
// Plane 2: Empty valid points (mask of 24 valid points minus occupied)
// Plane 3: Structural edges (horizontal connectors)
// Plane 4: Structural edges (vertical connectors)
void State::ObservationTensor(int /*player*/, float* out) const {
    std::memset(out, 0, sizeof(float) * kObservationSize);
    const int plane_stride = kObservationRows * kObservationCols;

    // Plane 0 & 1: pieces.
    for (int pos = 0; pos < kNumPositions; ++pos) {
        const int r = kBoardRow[pos];
        const int c = kBoardCol[pos];
        const int offset = r * kObservationCols + c;
        if ((board_[0] >> pos) & 1) out[0 * plane_stride + offset] = 1.0f;
        else if ((board_[1] >> pos) & 1) out[1 * plane_stride + offset] = 1.0f;
        else out[2 * plane_stride + offset] = 1.0f;  // empty valid point
    }

    // Plane 3 & 4: structural masks (constant).
    for (int r = 0; r < kObservationRows; ++r) {
        for (int c = 0; c < kObservationCols; ++c) {
            if (kObsPlane3Mask[r][c])
                out[3 * plane_stride + r * kObservationCols + c] = 1.0f;
            if (kObsPlane4Mask[r][c])
                out[4 * plane_stride + r * kObservationCols + c] = 1.0f;
        }
    }
}

// ----- Serialization (simple: one action per line, like OpenSpiel).
std::string State::Serialize() const {
    std::ostringstream os;
    for (int a : history_) os << a << "\n";
    return os.str();
}

State Game::Deserialize(const std::string& data) {
    State s;
    std::istringstream is(data);
    std::string line;
    while (std::getline(is, line)) {
        if (line.empty()) continue;
        s.ApplyAction(std::stoi(line));
    }
    return s;
}

// Simple fast xorshift64 RNG.
namespace {
inline uint64_t XorShift64(uint64_t& x) {
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    return x;
}

// Bitboard popcount helper.
inline int PopCount(BitBoard b) { return __builtin_popcount(b); }

// How many mills does `player` currently have on board?
inline int MillCount(const State& s, int player) {
    const BitBoard b = s.Board(player);
    int n = 0;
    for (int i = 0; i < 16; ++i) {
        if ((b & kMillMasks[i]) == kMillMasks[i]) ++n;
    }
    return n;
}
}  // namespace

int RandomAction(const State& s, uint64_t& rng) {
    int buf[256];
    const int n = s.LegalActionsInto(buf);
    if (n <= 0) return -1;
    return buf[XorShift64(rng) % static_cast<uint64_t>(n)];
}

// -----------------------------------------------------------------------
// Evaluation heuristic.
//
// Score is from the perspective of the *current* player of `s`.
// Components (all differences are own - opponent):
//   material      -- on-board + unplaced pieces
//   mills         -- completed 3-in-a-row lines
//   open_mills    -- mill threats (2-own + reachable empty slot)
//   double_mill   -- bonus when multiple simultaneous threats exist
//   blocked       -- own pieces with zero legal moves (moving phase only)
//   mobility      -- number of legal moves (approx)
// Weights shift in the flying endgame so mill threats dominate over
// raw mobility (which is meaningless when a piece can jump anywhere).
// Terminal wins return kWinScore; terminal losses return -kWinScore.
// Draws return 0.
// -----------------------------------------------------------------------
namespace {
constexpr int kWinScore = 100000;

// Count a player's (approximate) mobility: number of (from, to) or
// placement actions available to them *if it were their turn*. Cheap
// version: count own pieces * empty-adjacent for moving phase,
// or empties for placing phase.
int ApproxMobility(const State& s, int player) {
    const BitBoard own = s.Board(player);
    const BitBoard occ = s.Board(0) | s.Board(1);
    const BitBoard empty = (~occ) & kFullBoard;

    // If any pieces left to place, placement candidates = empty points.
    int mob = 0;
    if (s.MenToDeploy(player) > 0) {
        mob += PopCount(empty);
    }
    // Movement / flying mobility.
    const bool flying = (s.MenOnBoard(player) == 3) &&
                        (s.MenToDeploy(player) == 0);
    if (flying) {
        mob += PopCount(own) * PopCount(empty);
    } else {
        BitBoard pieces = own;
        while (pieces) {
            const int pos = __builtin_ctz(pieces);
            pieces &= pieces - 1;
            mob += PopCount(kAdjacency[pos] & empty);
        }
    }
    return mob;
}

// Count "open mill" threats for `player`: mill lines with 2 own stones,
// zero opponent stones, and an empty third slot reachable in one move.
//   - Placing phase: any empty slot is reachable (drop a stone there).
//   - Flying phase:  any empty slot is reachable (jump any own piece).
//   - Moving phase:  empty slot must be adjacent to an own piece that is
//                    not already inside this mill line.
int CountOpenMills(const State& s, int player) {
    const BitBoard own = s.Board(player);
    const BitBoard opp = s.Board(1 - player);
    const BitBoard occ = own | opp;
    const BitBoard empty = (~occ) & kFullBoard;

    const bool placing = s.MenToDeploy(player) > 0;
    const bool flying  = (s.MenOnBoard(player) == 3) && !placing;
    const bool easy_reach = placing || flying;

    int threats = 0;
    for (int i = 0; i < 16; ++i) {
        const BitBoard m = kMillMasks[i];
        const BitBoard own_in_line = own & m;
        if (PopCount(own_in_line) != 2) continue;
        if ((opp & m) != 0) continue;
        const BitBoard slot = m & empty;
        if (!slot) continue;
        if (easy_reach) {
            ++threats;
        } else {
            const int empty_pos = __builtin_ctz(slot);
            const BitBoard movers = kAdjacency[empty_pos] & own & ~m;
            if (movers) ++threats;
        }
    }
    return threats;
}

// Count "mill blocks": mill lines that hold exactly 2 opponent stones
// and 1 own stone. That own stone is the only thing preventing the
// opponent from forming the mill -- defensively very valuable.
int CountMillBlocks(const State& s, int player) {
    const BitBoard own = s.Board(player);
    const BitBoard opp = s.Board(1 - player);
    int count = 0;
    for (int i = 0; i < 16; ++i) {
        const BitBoard m = kMillMasks[i];
        if (PopCount(opp & m) == 2 && PopCount(own & m) == 1) ++count;
    }
    return count;
}

// Count "running mill" / swing-mill setups: pieces sitting inside a
// completed own mill that can be moved into an empty adjacent slot
// forming a *different* own mill. Each swing captures, so this pattern
// is a sustained threat -- far stronger than a one-shot mill.
//
// Only meaningful in the moving phase: in placing phase there are no
// "moves yet"; in flying phase every piece reaches every empty cell so
// the basic open-mill count already captures the threat.
int CountRunningMills(const State& s, int player) {
    const bool placing = s.MenToDeploy(player) > 0;
    const bool flying  = (s.MenOnBoard(player) == 3) && !placing;
    if (placing || flying) return 0;

    const BitBoard own = s.Board(player);
    const BitBoard occ = own | s.Board(1 - player);
    const BitBoard empty = (~occ) & kFullBoard;

    // Pieces currently inside a completed own mill.
    BitBoard in_mill = 0;
    for (int i = 0; i < 16; ++i) {
        const BitBoard m = kMillMasks[i];
        if ((own & m) == m) in_mill |= m;
    }

    int count = 0;
    BitBoard pieces = in_mill;
    while (pieces) {
        const int pos = __builtin_ctz(pieces);
        pieces &= pieces - 1;
        BitBoard targets = kAdjacency[pos] & empty;
        while (targets) {
            const int to = __builtin_ctz(targets);
            targets &= targets - 1;
            // Does moving `pos` -> `to` form a mill not containing `pos`?
            // Two candidate mills for `to`; skip the one containing `pos`.
            const BitBoard m1 = kPosMill1[to];
            const BitBoard m2 = kPosMill2[to];
            const BitBoard piece_bit = 1u << pos;
            const BitBoard to_bit    = 1u << to;
            const BitBoard own_minus = own & ~piece_bit;
            bool swings = false;
            if (m1 && !(m1 & piece_bit)) {
                if (((own_minus | to_bit) & m1) == m1) swings = true;
            }
            if (!swings && m2 && !(m2 & piece_bit)) {
                if (((own_minus | to_bit) & m2) == m2) swings = true;
            }
            if (swings) { ++count; break; }
        }
    }
    return count;
}

// Count own pieces that have no legal move (all adjacent squares
// occupied). Only meaningful in the moving phase -- flying pieces are
// never blocked, and placing-phase pieces can still be added elsewhere.
int CountBlockedPieces(const State& s, int player) {
    const bool placing = s.MenToDeploy(player) > 0;
    const bool flying  = (s.MenOnBoard(player) == 3) && !placing;
    if (placing || flying) return 0;

    const BitBoard own   = s.Board(player);
    const BitBoard occ   = own | s.Board(1 - player);
    const BitBoard empty = (~occ) & kFullBoard;

    int blocked = 0;
    BitBoard pieces = own;
    while (pieces) {
        const int pos = __builtin_ctz(pieces);
        pieces &= pieces - 1;
        if ((kAdjacency[pos] & empty) == 0) ++blocked;
    }
    return blocked;
}

inline bool IsFlying(const State& s, int player) {
    return (s.MenOnBoard(player) == 3) && (s.MenToDeploy(player) == 0);
}

// -----------------------------------------------------------------------
// Python-parity helpers used ONLY by the AI / reward-shaping evaluator.
// They intentionally mirror the quirks of `utils._count_potential_mills`
// and `utils._get_mobility` so switching `extract_state_features` from
// Python loops to the C++ fast path is a bit-exact drop-in -- no
// mid-training reward-landscape shift for in-flight runs.
//
// Differences from `CountOpenMills` / `ApproxMobility` (which are tuned
// for minimax search and stay unchanged):
//   * potential-mills "easy reach" trigger:
//       Python: total_on_board < 18  OR  my_pieces == 3
//       AI-pa.: total_on_board < 18  OR  my_pieces == 3  (matched)
//       minimax: per-player MenToDeploy>0 OR MenOnBoard==3 && deploy==0
//   * mobility flying-phase cap:
//       Python: my_pieces <= 3 -> cap moves at 12, no adjacency
//       AI-pa.: same
//       minimax: no cap, scales with own*empty
// -----------------------------------------------------------------------
int CountAIOpenMills(const State& s, int player) {
    const BitBoard own = s.Board(player);
    const BitBoard opp = s.Board(1 - player);
    const BitBoard occ = own | opp;
    const BitBoard empty = (~occ) & kFullBoard;

    const int my_pieces = PopCount(own);
    const int total_on_board = PopCount(occ);
    // Mirror Python: placement-phase check by *total* board occupancy
    // against the magic "18", and flying-phase check by own piece count.
    const bool any_empty_counts = (total_on_board < 18) || (my_pieces == 3);

    int count = 0;
    for (int i = 0; i < 16; ++i) {
        const BitBoard m = kMillMasks[i];
        if (PopCount(own & m) != 2) continue;
        if ((opp & m) != 0)         continue;
        const BitBoard slot = m & empty;
        if (PopCount(slot) != 1)    continue;

        if (any_empty_counts) {
            ++count;
            continue;
        }
        const int empty_pos = __builtin_ctz(slot);
        const BitBoard movers = kAdjacency[empty_pos] & own & ~m;
        if (movers) ++count;
    }
    return count;
}

int CountAIMobility(const State& s, int player) {
    const BitBoard own = s.Board(player);
    const BitBoard occ = s.Board(0) | s.Board(1);
    const BitBoard empty = (~occ) & kFullBoard;

    const int my_pieces = PopCount(own);
    int moves = 0;
    BitBoard pieces = own;
    while (pieces) {
        const int pos = __builtin_ctz(pieces);
        pieces &= pieces - 1;
        if (my_pieces <= 3) {
            // Python: any empty target counts; cap at 12 total to keep
            // the shaping feature bounded in the flying endgame.
            moves += PopCount(empty);
            if (moves >= 12) return 12;
        } else {
            moves += PopCount(kAdjacency[pos] & empty);
        }
    }
    return moves;
}
}  // namespace

// Fast path: hardcoded const weights. Lets the compiler fold every
// weight into an immediate, which the alpha-beta hot loop relies on.
// Used by MinimaxSearch (free function) and by MinimaxEngine when no
// custom weights have been installed via SetWeights().
int Evaluate(const State& s) {
    if (s.IsTerminal()) {
        const int w = s.Winner();
        if (w == 2) return 0;
        return kWinScore;
    }
    const int cp  = s.CurrentPlayer();
    const int opp = 1 - cp;

    const int own_material = s.MenOnBoard(cp)  + s.MenToDeploy(cp);
    const int opp_material = s.MenOnBoard(opp) + s.MenToDeploy(opp);
    const int own_mills    = MillCount(s, cp);
    const int opp_mills    = MillCount(s, opp);
    const int own_open     = CountOpenMills(s, cp);
    const int opp_open     = CountOpenMills(s, opp);
    const int own_run      = CountRunningMills(s, cp);
    const int opp_run      = CountRunningMills(s, opp);
    const int own_mblk     = CountMillBlocks(s, cp);
    const int opp_mblk     = CountMillBlocks(s, opp);
    const int own_blocked  = CountBlockedPieces(s, cp);
    const int opp_blocked  = CountBlockedPieces(s, opp);
    const int own_mob      = ApproxMobility(s, cp);
    const int opp_mob      = ApproxMobility(s, opp);

    const bool endgame = IsFlying(s, cp) || IsFlying(s, opp);

    const int W_MAT  = endgame ? 14 :  8;
    const int W_MILL = endgame ? 22 : 18;
    const int W_OPEN = endgame ? 22 : 14;
    const int W_RUN  = 30;
    const int W_DBL  = 28;
    const int W_MBLK = 10;
    const int W_BLK  = endgame ?  0 : 10;
    const int W_MOB  = endgame ?  0 :  1;

    const int own_eff = own_open + own_run;
    const int opp_eff = opp_open + opp_run;
    const int own_double = (own_eff >= 2) ? (own_eff - 1) * W_DBL : 0;
    const int opp_double = (opp_eff >= 2) ? (opp_eff - 1) * W_DBL : 0;

    return W_MAT  * (own_material - opp_material)
         + W_MILL * (own_mills    - opp_mills)
         + W_OPEN * (own_open     - opp_open)
         + W_RUN  * (own_run      - opp_run)
         +          (own_double   - opp_double)
         + W_MBLK * (own_mblk     - opp_mblk)
         - W_BLK  * (own_blocked  - opp_blocked)
         + W_MOB  * (own_mob      - opp_mob);
}

// Tunable variant. Used only when MinimaxEngine.SetWeights() has been
// called -- pays a few extra memory loads per leaf vs the const path.
int Evaluate(const State& s, const EvalWeights& weights) {
    if (s.IsTerminal()) {
        const int w = s.Winner();
        if (w == 2) return 0;
        return kWinScore;
    }
    const int cp  = s.CurrentPlayer();
    const int opp = 1 - cp;

    const int own_material = s.MenOnBoard(cp)  + s.MenToDeploy(cp);
    const int opp_material = s.MenOnBoard(opp) + s.MenToDeploy(opp);
    const int own_mills    = MillCount(s, cp);
    const int opp_mills    = MillCount(s, opp);
    const int own_open     = CountOpenMills(s, cp);
    const int opp_open     = CountOpenMills(s, opp);
    const int own_run      = CountRunningMills(s, cp);
    const int opp_run      = CountRunningMills(s, opp);
    const int own_mblk     = CountMillBlocks(s, cp);
    const int opp_mblk     = CountMillBlocks(s, opp);
    const int own_blocked  = CountBlockedPieces(s, cp);
    const int opp_blocked  = CountBlockedPieces(s, opp);
    const int own_mob      = ApproxMobility(s, cp);
    const int opp_mob      = ApproxMobility(s, opp);

    const bool endgame = IsFlying(s, cp) || IsFlying(s, opp);

    const int W_MAT  = endgame ? weights.w_material_end : weights.w_material_mid;
    const int W_MILL = endgame ? weights.w_mill_end     : weights.w_mill_mid;
    const int W_OPEN = endgame ? weights.w_open_end     : weights.w_open_mid;
    const int W_RUN  = weights.w_running;
    const int W_DBL  = weights.w_double;
    const int W_MBLK = weights.w_mill_block;
    const int W_BLK  = endgame ? 0 : weights.w_blocked_mid;
    const int W_MOB  = endgame ? 0 : weights.w_mobility_mid;

    // Running mills also count toward the simultaneous-threat tally so
    // a "mill + running mill" position trips the double-mill bonus.
    const int own_eff = own_open + own_run;
    const int opp_eff = opp_open + opp_run;
    const int own_double = (own_eff >= 2) ? (own_eff - 1) * W_DBL : 0;
    const int opp_double = (opp_eff >= 2) ? (opp_eff - 1) * W_DBL : 0;

    return W_MAT  * (own_material - opp_material)
         + W_MILL * (own_mills    - opp_mills)
         + W_OPEN * (own_open     - opp_open)
         + W_RUN  * (own_run      - opp_run)
         +          (own_double   - opp_double)
         + W_MBLK * (own_mblk     - opp_mblk)
         - W_BLK  * (own_blocked  - opp_blocked)
         + W_MOB  * (own_mob      - opp_mob);
}

EvalBreakdown EvaluateBreakdown(const State& s) {
    EvalBreakdown b{};
    b.current_player = s.IsTerminal() ? -1 : s.CurrentPlayer();
    if (s.IsTerminal()) {
        b.total = Evaluate(s);
        return b;
    }
    const int cp  = s.CurrentPlayer();
    const int opp = 1 - cp;

    b.own_material = s.MenOnBoard(cp)  + s.MenToDeploy(cp);
    b.opp_material = s.MenOnBoard(opp) + s.MenToDeploy(opp);
    b.own_mills    = MillCount(s, cp);
    b.opp_mills    = MillCount(s, opp);
    b.own_open_mills = CountOpenMills(s, cp);
    b.opp_open_mills = CountOpenMills(s, opp);
    b.own_running_mills = CountRunningMills(s, cp);
    b.opp_running_mills = CountRunningMills(s, opp);
    b.own_mill_blocks   = CountMillBlocks(s, cp);
    b.opp_mill_blocks   = CountMillBlocks(s, opp);
    b.own_blocked  = CountBlockedPieces(s, cp);
    b.opp_blocked  = CountBlockedPieces(s, opp);
    b.own_mobility = ApproxMobility(s, cp);
    b.opp_mobility = ApproxMobility(s, opp);

    b.endgame = IsFlying(s, cp) || IsFlying(s, opp);

    b.w_material      = b.endgame ? 14 :  8;
    b.w_mill          = b.endgame ? 22 : 18;
    b.w_open_mill     = b.endgame ? 22 : 14;
    b.w_running_mill  = 30;
    b.w_double_mill   = 28;
    b.w_mill_block    = 10;
    b.w_blocked       = b.endgame ?  0 : 10;
    b.w_mobility      = b.endgame ?  0 :  1;

    const int own_eff = b.own_open_mills + b.own_running_mills;
    const int opp_eff = b.opp_open_mills + b.opp_running_mills;
    const int own_double = (own_eff >= 2)
        ? (own_eff - 1) * b.w_double_mill : 0;
    const int opp_double = (opp_eff >= 2)
        ? (opp_eff - 1) * b.w_double_mill : 0;

    b.material_score      = b.w_material     * (b.own_material      - b.opp_material);
    b.mill_score          = b.w_mill         * (b.own_mills         - b.opp_mills);
    b.open_mill_score     = b.w_open_mill    * (b.own_open_mills    - b.opp_open_mills);
    b.running_mill_score  = b.w_running_mill * (b.own_running_mills - b.opp_running_mills);
    b.double_mill_score   = own_double - opp_double;
    b.mill_block_score    = b.w_mill_block   * (b.own_mill_blocks   - b.opp_mill_blocks);
    b.blocked_score       = -b.w_blocked     * (b.own_blocked       - b.opp_blocked);
    b.mobility_score      = b.w_mobility     * (b.own_mobility      - b.opp_mobility);

    b.total = b.material_score + b.mill_score + b.open_mill_score
            + b.running_mill_score + b.double_mill_score
            + b.mill_block_score + b.blocked_score + b.mobility_score;
    return b;
}

// -----------------------------------------------------------------------
// AI / reward-shaping evaluator.
//
// Mirrors `utils.extract_state_features` from the Python side: same
// feature counts, but always from the perspective of the *given*
// `player`, not whoever happens to be the side-to-move. Reward shaping
// in worker.py calls this twice per AI move (pre- and post-apply), with
// `player` fixed to the AI for both calls -- so we cannot rely on
// CurrentPlayer().
//
// All field semantics are documented next to `AIEvalBreakdown` in the
// header. Helper functions live in the anonymous namespace just above.
// -----------------------------------------------------------------------
AIEvalBreakdown EvaluateForAIBreakdown(const State& s, int player) {
    AIEvalBreakdown b{};
    b.player = player;
    if (s.IsTerminal()) {
        // Terminal positions have no meaningful shaping signal; everything
        // stays zero and the Python side computes the terminal reward
        // separately from `state.returns()`.
        return b;
    }
    if (player != 0 && player != 1) {
        return b;
    }
    const int opp = 1 - player;

    // Python parity: count only stones on the board, not unplaced reserves.
    // (utils._count_pieces walks the parsed 24-tuple board; reserves don't
    // appear there.)
    b.my_pieces  = s.MenOnBoard(player);
    b.opp_pieces = s.MenOnBoard(opp);

    b.my_mills  = MillCount(s, player);
    b.opp_mills = MillCount(s, opp);

    b.my_potential_mills  = CountAIOpenMills(s, player);
    b.opp_potential_mills = CountAIOpenMills(s, opp);

    b.my_blocked_mills  = CountMillBlocks(s, player);
    b.opp_blocked_mills = CountMillBlocks(s, opp);

    // "Unblocked threats" from `player`'s view is the count of mills the
    // *opponent* can close next turn -- i.e. opp's potential mills.
    b.my_unblocked_threats  = b.opp_potential_mills;
    b.opp_unblocked_threats = b.my_potential_mills;

    // Python "double_mills" semantics: pieces inside a completed own mill
    // that can swing into an adjacent empty slot forming a *different*
    // own mill. That is exactly what CountRunningMills computes.
    b.my_double_mills  = CountRunningMills(s, player);
    b.opp_double_mills = CountRunningMills(s, opp);

    b.my_mobility  = CountAIMobility(s, player);
    b.opp_mobility = CountAIMobility(s, opp);

    b.endgame = IsFlying(s, player) || IsFlying(s, opp);
    return b;
}

int EvaluateForAI(const State& s, int player) {
    // Default integer-weight scoring intended as a fast standalone signal
    // when the curriculum-tunable Python potential is overkill. The Python
    // side does not call this; it uses EvaluateForAIBreakdown and combines
    // raw counts with the curriculum's float weights (which can be scaled
    // by `shaping_multiplier`). Weights are chosen to roughly match the
    // *relative* magnitudes in default reward_config (×1000) so the score
    // is comparable across feature components without floats.
    const AIEvalBreakdown b = EvaluateForAIBreakdown(s, player);
    if (b.player < 0) return 0;

    constexpr int W_MILL        = 300;   // mill_reward
    constexpr int W_OPEN        = 200;   // setup_capture_reward
    constexpr int W_OPP_OPEN    = 200;   // block_mill_reward (against)
    constexpr int W_DOUBLE      = 500;   // double_mill_reward
    constexpr int W_MATERIAL    = 20;    // piece_advantage_reward
    constexpr int W_MOBILITY    = 50;    // mobility_reward

    return W_MILL     * (b.my_mills          - b.opp_mills)
         + W_OPEN     *  b.my_potential_mills
         - W_OPP_OPEN *  b.my_unblocked_threats
         + W_DOUBLE   *  b.my_double_mills
         + W_MATERIAL * (b.my_pieces          - b.opp_pieces)
         + W_MOBILITY * (b.my_mobility        - b.opp_mobility);
}

// -----------------------------------------------------------------------
// Negamax with alpha-beta pruning.
//
// Conceptually for 2-player zero-sum games; Nine Men's Morris has the
// subtlety that a "capture" is not a change of turn -- the same player
// moves again. We handle this by negating only when the next state has
// a different current_player than the current one.
//
// Returns score from the perspective of s.CurrentPlayer().
// -----------------------------------------------------------------------
namespace {
int Negamax(const State& s, int depth, int alpha, int beta,
            long& nodes_visited) {
    ++nodes_visited;

    if (s.IsTerminal()) {
        if (s.Winner() == 2) return 0;  // Draw.
        // The side that just moved always won (a move can only end the
        // game in the mover's favor), so +kWinScore means "mover won";
        // the caller's negation converts this to its own perspective.
        return +kWinScore;
    }

    if (depth == 0) {
        return Evaluate(s);
    }

    int buf[256];
    const int n = s.LegalActionsInto(buf);
    if (n == 0) {
        // Shouldn't happen -- non-terminal with no actions implies we
        // missed a stalemate. Treat as loss for side-to-move.
        return -kWinScore;
    }

    int best = -kWinScore - 1;
    const int cur_player = s.CurrentPlayer();

    for (int i = 0; i < n; ++i) {
        State child = s;
        child.ApplyAction(buf[i]);

        int child_score;
        if (child.IsTerminal()) {
            const int w = child.Winner();
            if (w == 2) {
                child_score = 0;
            } else {
                // The side that just moved is `cur_player`.
                // If winner == cur_player => +kWinScore (good for us),
                // else -kWinScore.
                child_score = (w == cur_player) ? +kWinScore : -kWinScore;
                // Depth tie-break: prefer faster wins, slower losses.
                child_score -= (child_score > 0 ?  (100 - depth)
                                                 : -(100 - depth));
            }
        } else if (child.CurrentPlayer() == cur_player) {
            // Same player moves again (we just formed a mill -> capture).
            // Recurse WITHOUT negating.
            child_score = Negamax(child, depth - 1, alpha, beta, nodes_visited);
        } else {
            // Turn switched: standard negamax negation.
            child_score = -Negamax(child, depth - 1, -beta, -alpha, nodes_visited);
        }

        if (child_score > best)  best  = child_score;
        if (best        > alpha) alpha = best;
        if (alpha       >= beta) break;  // alpha-beta cutoff
    }
    return best;
}
}  // namespace

SearchResult MinimaxSearch(const State& s, int depth) {
    if (s.IsTerminal()) {
        return {-1, 0, 0};
    }
    int buf[256];
    const int n = s.LegalActionsInto(buf);
    if (n == 0) return {-1, -kWinScore, 0};

    long nodes = 0;
    int best_action = buf[0];
    int best_score  = -kWinScore - 1;
    int alpha = -kWinScore - 1;
    const int beta = kWinScore + 1;
    const int cur_player = s.CurrentPlayer();

    for (int i = 0; i < n; ++i) {
        State child = s;
        child.ApplyAction(buf[i]);

        int child_score;
        if (child.IsTerminal()) {
            const int w = child.Winner();
            if (w == 2) {
                child_score = 0;
            } else {
                child_score = (w == cur_player) ? +kWinScore : -kWinScore;
                child_score -= (child_score > 0 ?  (100 - depth)
                                                 : -(100 - depth));
            }
        } else if (child.CurrentPlayer() == cur_player) {
            child_score = Negamax(child, depth - 1, alpha, beta, nodes);
        } else {
            child_score = -Negamax(child, depth - 1, -beta, -alpha, nodes);
        }

        if (child_score > best_score) {
            best_score  = child_score;
            best_action = buf[i];
        }
        if (best_score > alpha) alpha = best_score;
    }
    return {best_action, best_score, nodes};
}

double RandomPlayouts(int num_games, uint64_t seed, int* out_lengths) {
    uint64_t rng = seed ? seed : 0xdeadbeefcafebabeULL;
    double sum_returns_p0 = 0.0;
    int buf[256];
    for (int g = 0; g < num_games; ++g) {
        State s;
        int actions = 0;
        while (!s.IsTerminal()) {
            const int n = s.LegalActionsInto(buf);
            // n is guaranteed > 0 in non-terminal states.
            const int pick = static_cast<int>(XorShift64(rng) % static_cast<uint64_t>(n));
            s.ApplyAction(buf[pick]);
            ++actions;
        }
        sum_returns_p0 += s.Returns()[0];
        if (out_lengths) out_lengths[g] = actions;
    }
    return sum_returns_p0;
}

// =========================================================================
// Zobrist hashing.
//
// Position state is fully described by:
//   board_[0]  (24 bits)            -- white pieces on the board
//   board_[1]  (24 bits)            -- black pieces on the board
//   unplaced_[0..1]  (4 bits each)  -- 0..9 stones left to deploy
//   current_player_  (1 bit)
//   phase_           (capture vs not -- placing/moving derived per-player)
//   turn_            (matters because of max_turns_ draw cap)
//
// Each of these mixes a fixed 64-bit random value into the key; identical
// state -> identical key, and any single-component change flips many bits.
// =========================================================================
namespace {

struct ZobristTables {
    uint64_t piece[kNumPositions][kNumPlayers];   // 24 * 2
    uint64_t unplaced[kNumPlayers][kPiecesPerPlayer + 1];   // 0..9
    uint64_t side_to_move;
    uint64_t capture_phase;
    uint64_t turn[kMaxGameLength + 1];
};

inline uint64_t SplitMix64(uint64_t& s) {
    s += 0x9E3779B97F4A7C15ULL;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

const ZobristTables& Zobrist() {
    static const ZobristTables t = [] {
        ZobristTables out;
        // Deterministic seed -> reproducible keys across processes / runs.
        uint64_t s = 0xC0FFEE1234567890ULL;
        for (int p = 0; p < kNumPositions; ++p) {
            for (int c = 0; c < kNumPlayers; ++c) {
                out.piece[p][c] = SplitMix64(s);
            }
        }
        for (int c = 0; c < kNumPlayers; ++c) {
            for (int u = 0; u <= kPiecesPerPlayer; ++u) {
                out.unplaced[c][u] = SplitMix64(s);
            }
        }
        out.side_to_move  = SplitMix64(s);
        out.capture_phase = SplitMix64(s);
        for (int t = 0; t <= kMaxGameLength; ++t) {
            out.turn[t] = SplitMix64(s);
        }
        return out;
    }();
    return t;
}

}  // namespace

uint64_t ZobristKey(const State& s) {
    const auto& z = Zobrist();
    uint64_t h = 0;

    for (int color = 0; color < kNumPlayers; ++color) {
        BitBoard b = s.Board(color);
        while (b) {
            const int pos = __builtin_ctz(b);
            b &= b - 1;
            h ^= z.piece[pos][color];
        }
        const int u = s.MenToDeploy(color);
        h ^= z.unplaced[color][u];
    }
    if (s.CurrentPlayer() == kBlack) h ^= z.side_to_move;
    if (s.CurrentPhase() == Phase::kCapture) h ^= z.capture_phase;
    int t = s.Turn();
    if (t < 0) t = 0;
    if (t > kMaxGameLength) t = kMaxGameLength;
    h ^= z.turn[t];
    return h;
}

// =========================================================================
// MinimaxEngine: TT-backed alpha-beta.
//
// Two modes:
//   - strict (default): bit-exact match to MinimaxSearch. TT probe
//     requires cached_depth == requested_depth.
//   - relaxed (opt-in): mate-distance scoring (chess-engine convention)
//     + cached_depth >= requested_depth probe. Non-mate move selection
//     identical to strict; mate-distance tie-breaking can differ. In
//     normal forward minimax the hit rate matches strict (see the header
//     comment on the constructor).
// =========================================================================
namespace {

constexpr int kMaxLegalActions = 256;

// In relaxed mode we use position-relative mate-distance scores:
//   sign * (kRelaxedMate - distance_in_plies_to_terminal)
// distance==1 => the move *is* the mating move; deeper distances
// are smaller-magnitude. kRelaxedMate is chosen so the entire mate
// range fits in (kMateThreshold .. kRelaxedMate]; values below
// kMateThreshold are "non-mate" (heuristic eval).
constexpr int kRelaxedMate     = 30000;
constexpr int kMateThreshold   = 29000;
constexpr int kMaxMateDistance = kRelaxedMate - kMateThreshold;

inline bool IsMateScore(int s) { return s > kMateThreshold || s < -kMateThreshold; }

inline int AdjustForOnePlyDeeper(int s) {
    // Move the mate-distance one ply farther (we're one step away from
    // where the score was computed). Magnitude shrinks by 1.
    if (s >  kMateThreshold) return s - 1;
    if (s < -kMateThreshold) return s + 1;
    return s;
}

}  // namespace

MinimaxEngine::MinimaxEngine(std::size_t tt_bytes, bool strict_parity)
    : strict_(strict_parity) {
    Allocate(tt_bytes);
}

MinimaxEngine::~MinimaxEngine() {
    delete[] table_;
}

void MinimaxEngine::Allocate(std::size_t tt_bytes) {
    // Round down to power-of-two number of entries. Minimum 1024 entries.
    std::size_t entries = tt_bytes / sizeof(TTEntry);
    if (entries < 1024) entries = 1024;
    // Round down to power of two.
    std::size_t pot = 1;
    while ((pot << 1) <= entries) pot <<= 1;
    num_entries_ = pot;
    mask_ = static_cast<uint64_t>(pot - 1);

    table_ = new TTEntry[num_entries_];
    // Zero-init: all keys 0, depths 0, flags 0. Generation starts at 1 so
    // a stale 0-generation slot (never written) is never confused for a
    // current-gen entry on the rare occasion key 0 collides.
    std::memset(table_, 0, sizeof(TTEntry) * num_entries_);
    generation_ = 1;
    probes_ = hits_ = stores_ = collisions_ = 0;
}

std::size_t MinimaxEngine::TtBytes() const {
    return num_entries_ * sizeof(TTEntry);
}

void MinimaxEngine::TtClear() {
    std::memset(table_, 0, sizeof(TTEntry) * num_entries_);
    generation_ = 1;
    probes_ = hits_ = stores_ = collisions_ = 0;
}

void MinimaxEngine::NewGame() {
    // 6-bit generation counter (bits 2..7 of flag_age).
    generation_ = static_cast<uint8_t>((generation_ + 1) & 0x3F);
    if (generation_ == 0) generation_ = 1;
}

double MinimaxEngine::TtFillFraction() const {
    if (num_entries_ == 0) return 0.0;
    // Sample at most 16K slots for a quick estimate.
    const std::size_t sample = std::min<std::size_t>(num_entries_, 16384);
    std::size_t filled = 0;
    const std::size_t step = num_entries_ / sample;
    for (std::size_t i = 0; i < sample; ++i) {
        const TTEntry& e = table_[i * step];
        if (e.key != 0 || e.depth != 0 || e.flag_age != 0) ++filled;
    }
    return static_cast<double>(filled) / static_cast<double>(sample);
}

bool MinimaxEngine::ProbeStrict(uint64_t key, int depth, int alpha, int beta,
                                int& out_score, int& out_move) const {
    ++probes_;
    const TTEntry& e = table_[Index(key)];
    if (e.key != key || e.flag_age == 0) {
        out_move = -1;
        return false;
    }
    out_move = e.best_move;
    if (e.depth != depth) return false;
    const uint8_t flag = e.flag_age & 0x3;
    if (flag == kFlagExact) {
        out_score = e.score;
        ++hits_;
        return true;
    }
    if (flag == kFlagLower && e.score >= beta) {
        out_score = e.score;
        ++hits_;
        return true;
    }
    if (flag == kFlagUpper && e.score <= alpha) {
        out_score = e.score;
        ++hits_;
        return true;
    }
    return false;
}

bool MinimaxEngine::ProbeRelaxed(uint64_t key, int depth, int alpha, int beta,
                                 int& out_score, int& out_move) const {
    ++probes_;
    const TTEntry& e = table_[Index(key)];
    if (e.key != key || e.flag_age == 0) {
        out_move = -1;
        return false;
    }
    out_move = e.best_move;
    // Relaxed: cached_depth >= requested suffices. Score is in
    // mate-distance encoding which is position-relative, so reusing a
    // deeper cached score for a shallower query gives the parent a
    // strictly more accurate child value.
    if (e.depth < depth) return false;
    const uint8_t flag = e.flag_age & 0x3;
    if (flag == kFlagExact) {
        out_score = e.score;
        ++hits_;
        return true;
    }
    if (flag == kFlagLower && e.score >= beta) {
        out_score = e.score;
        ++hits_;
        return true;
    }
    if (flag == kFlagUpper && e.score <= alpha) {
        out_score = e.score;
        ++hits_;
        return true;
    }
    return false;
}

void MinimaxEngine::Store(uint64_t key, int depth, int score, int move,
                          Flag flag) {
    TTEntry& e = table_[Index(key)];
    const uint8_t old_gen = (e.flag_age >> 2) & 0x3F;
    const bool slot_empty = (e.flag_age == 0 && e.key == 0);
    const bool stale      = !slot_empty && (old_gen != generation_);
    const bool deeper     = (depth >= e.depth);
    const bool same_key   = (e.key == key);

    bool replace = slot_empty || same_key || stale || deeper;
    if (!replace) {
        ++collisions_;
        return;
    }
    if (!same_key && !slot_empty) {
        ++collisions_;
    }

    e.key       = key;
    e.score     = static_cast<int32_t>(score);
    e.best_move = static_cast<int16_t>(move);
    e.depth     = static_cast<int8_t>(depth);
    e.flag_age  = static_cast<uint8_t>(
        (static_cast<uint8_t>(flag) & 0x3) |
        (static_cast<uint8_t>(generation_ & 0x3F) << 2));
    ++stores_;
}

// ---- Strict mode: matches MinimaxSearch bit-exactly. ---------------------
int MinimaxEngine::NegamaxStrict(const State& s, int depth, int alpha, int beta,
                                 long& nodes, uint64_t key) {
    ++nodes;

    if (s.IsTerminal()) {
        const int w = s.Winner();
        if (w == 2) return 0;
        return +kWinScore;
    }
    if (depth == 0) {
        return use_custom_weights_ ? Evaluate(s, weights_) : Evaluate(s);
    }

    const int alpha_orig = alpha;

    int tt_score = 0, tt_move = -1;
    if (ProbeStrict(key, depth, alpha, beta, tt_score, tt_move)) {
        return tt_score;
    }

    int buf[kMaxLegalActions];
    const int n = s.LegalActionsInto(buf);
    if (n == 0) return -kWinScore;

    int order[kMaxLegalActions];
    int n_ord = 0;
    if (tt_move >= 0) {
        for (int i = 0; i < n; ++i) {
            if (buf[i] == tt_move) {
                order[n_ord++] = buf[i];
                buf[i] = -1;
                break;
            }
        }
    }
    for (int i = 0; i < n; ++i) if (buf[i] >= 0) order[n_ord++] = buf[i];

    int best_score = -kWinScore - 1;
    int best_move  = order[0];
    const int cur_player = s.CurrentPlayer();

    for (int i = 0; i < n_ord; ++i) {
        const int action = order[i];
        State child = s;
        child.ApplyAction(action);

        int child_score;
        if (child.IsTerminal()) {
            const int w = child.Winner();
            if (w == 2) {
                child_score = 0;
            } else {
                child_score = (w == cur_player) ? +kWinScore : -kWinScore;
                child_score -= (child_score > 0 ?  (100 - depth)
                                                 : -(100 - depth));
            }
        } else if (child.CurrentPlayer() == cur_player) {
            const uint64_t child_key = ZobristKey(child);
            child_score = NegamaxStrict(child, depth - 1, alpha, beta, nodes, child_key);
        } else {
            const uint64_t child_key = ZobristKey(child);
            child_score = -NegamaxStrict(child, depth - 1, -beta, -alpha, nodes, child_key);
        }

        if (child_score > best_score) { best_score = child_score; best_move = action; }
        if (best_score > alpha) alpha = best_score;
        if (alpha >= beta) break;
    }

    Flag flag;
    if (best_score <= alpha_orig)      flag = kFlagUpper;
    else if (best_score >= beta)       flag = kFlagLower;
    else                               flag = kFlagExact;
    Store(key, depth, best_score, best_move, flag);
    return best_score;
}

// ---- Relaxed mode: position-relative mate-distance + cross-search reuse. -
// Mate scores: |s| in (kMateThreshold, kRelaxedMate] with magnitude
// = kRelaxedMate - distance_in_plies. Smaller distance => larger
// magnitude. Distance is incremented automatically as scores are
// propagated up (AdjustForOnePlyDeeper).
int MinimaxEngine::NegamaxRelaxed(const State& s, int depth, int alpha, int beta,
                                  long& nodes, uint64_t key) {
    ++nodes;

    if (s.IsTerminal()) {
        const int w = s.Winner();
        if (w == 2) return 0;
        // Already terminal -- distance 0 to terminal. The PARENT will
        // adjust to "mate in 1" via AdjustForOnePlyDeeper after recursion.
        // (NegamaxRelaxed is normally called only on non-terminal states
        // because parents check IsTerminal first and short-circuit; this
        // branch is just defensive.)
        return +kRelaxedMate;
    }
    if (depth == 0) {
        return use_custom_weights_ ? Evaluate(s, weights_) : Evaluate(s);
    }

    const int alpha_orig = alpha;

    int tt_score = 0, tt_move = -1;
    if (ProbeRelaxed(key, depth, alpha, beta, tt_score, tt_move)) {
        return tt_score;
    }

    int buf[kMaxLegalActions];
    const int n = s.LegalActionsInto(buf);
    if (n == 0) return -kRelaxedMate;

    int order[kMaxLegalActions];
    int n_ord = 0;
    if (tt_move >= 0) {
        for (int i = 0; i < n; ++i) {
            if (buf[i] == tt_move) {
                order[n_ord++] = buf[i];
                buf[i] = -1;
                break;
            }
        }
    }
    for (int i = 0; i < n; ++i) if (buf[i] >= 0) order[n_ord++] = buf[i];

    int best_score = -kRelaxedMate - 1;
    int best_move  = order[0];
    const int cur_player = s.CurrentPlayer();

    for (int i = 0; i < n_ord; ++i) {
        const int action = order[i];
        State child = s;
        child.ApplyAction(action);

        int child_score;
        if (child.IsTerminal()) {
            const int w = child.Winner();
            if (w == 2) {
                child_score = 0;
            } else {
                // Mate in 1 ply (the move just made was the mating move).
                child_score = (w == cur_player) ?
                                 +(kRelaxedMate - 1) : -(kRelaxedMate - 1);
            }
        } else if (child.CurrentPlayer() == cur_player) {
            const uint64_t child_key = ZobristKey(child);
            child_score = NegamaxRelaxed(child, depth - 1, alpha, beta, nodes, child_key);
            child_score = AdjustForOnePlyDeeper(child_score);
        } else {
            const uint64_t child_key = ZobristKey(child);
            child_score = -NegamaxRelaxed(child, depth - 1, -beta, -alpha, nodes, child_key);
            child_score = AdjustForOnePlyDeeper(child_score);
        }

        if (child_score > best_score) { best_score = child_score; best_move = action; }
        if (best_score > alpha) alpha = best_score;
        if (alpha >= beta) break;
    }

    Flag flag;
    if (best_score <= alpha_orig)      flag = kFlagUpper;
    else if (best_score >= beta)       flag = kFlagLower;
    else                               flag = kFlagExact;
    Store(key, depth, best_score, best_move, flag);
    return best_score;
}

SearchResult MinimaxEngine::Search(const State& s, int depth) {
    if (s.IsTerminal()) return {-1, 0, 0};
    int buf[kMaxLegalActions];
    const int n = s.LegalActionsInto(buf);
    if (n == 0) {
        return {-1, strict_ ? -kWinScore : -kRelaxedMate, 0};
    }

    long nodes = 0;
    int best_action = buf[0];
    const int max_mag = strict_ ? kWinScore : kRelaxedMate;
    int best_score  = -max_mag - 1;
    int alpha = -max_mag - 1;
    const int beta = max_mag + 1;
    const int cur_player = s.CurrentPlayer();
    const uint64_t root_key = ZobristKey(s);

    for (int i = 0; i < n; ++i) {
        State child = s;
        child.ApplyAction(buf[i]);

        int child_score;
        if (child.IsTerminal()) {
            const int w = child.Winner();
            if (w == 2) {
                child_score = 0;
            } else if (strict_) {
                child_score = (w == cur_player) ? +kWinScore : -kWinScore;
                child_score -= (child_score > 0 ?  (100 - depth)
                                                 : -(100 - depth));
            } else {
                child_score = (w == cur_player) ?
                                 +(kRelaxedMate - 1) : -(kRelaxedMate - 1);
            }
        } else if (child.CurrentPlayer() == cur_player) {
            const uint64_t child_key = ZobristKey(child);
            if (strict_) {
                child_score = NegamaxStrict(child, depth - 1, alpha, beta, nodes, child_key);
            } else {
                child_score = NegamaxRelaxed(child, depth - 1, alpha, beta, nodes, child_key);
                child_score = AdjustForOnePlyDeeper(child_score);
            }
        } else {
            const uint64_t child_key = ZobristKey(child);
            if (strict_) {
                child_score = -NegamaxStrict(child, depth - 1, -beta, -alpha, nodes, child_key);
            } else {
                child_score = -NegamaxRelaxed(child, depth - 1, -beta, -alpha, nodes, child_key);
                child_score = AdjustForOnePlyDeeper(child_score);
            }
        }

        if (child_score > best_score) {
            best_score  = child_score;
            best_action = buf[i];
        }
        if (best_score > alpha) alpha = best_score;
        // No alpha-beta cutoff at the root.
    }

    Store(root_key, depth, best_score, best_action, kFlagExact);
    return {best_action, best_score, nodes};
}

int MinimaxEngine::Eval(const State& s) const {
    return use_custom_weights_ ? Evaluate(s, weights_) : Evaluate(s);
}

void MinimaxEngine::SetWeights(const EvalWeights& w) {
    weights_ = w;
    use_custom_weights_ = true;
    // Cached scores were computed under the old weights, so they must
    // not be reused. Wipe the table outright (also resets stats).
    TtClear();
}

// =========================================================================
// PlayUntilPlayer.
// =========================================================================
std::vector<int> PlayUntilPlayer(State& state,
                                 int target_player,
                                 const OpponentSpec& opp,
                                 MinimaxEngine* engine,
                                 uint64_t& rng_state) {
    std::vector<int> taken;
    taken.reserve(8);

    int buf[kMaxLegalActions];

    while (!state.IsTerminal() && state.CurrentPlayer() != target_player) {
        int action = -1;

        if (opp.kind == OpponentKind::kRandom) {
            const int n = state.LegalActionsInto(buf);
            if (n <= 0) break;
            action = buf[XorShift64(rng_state) % static_cast<uint64_t>(n)];
        } else {
            // Minimax. Optionally roll for an exploration random move.
            if (opp.random_move_prob > 0.0) {
                // Convert xorshift64 output to a [0, 1) double and compare.
                const uint64_t r = XorShift64(rng_state);
                const double u = static_cast<double>(r >> 11) /
                                 static_cast<double>(1ULL << 53);
                if (u < opp.random_move_prob) {
                    const int n = state.LegalActionsInto(buf);
                    if (n <= 0) break;
                    action = buf[XorShift64(rng_state) %
                                 static_cast<uint64_t>(n)];
                }
            }
            if (action < 0) {
                if (engine != nullptr) {
                    SearchResult r = engine->Search(state, opp.minimax_depth);
                    action = r.action;
                } else {
                    SearchResult r = MinimaxSearch(state, opp.minimax_depth);
                    action = r.action;
                }
            }
        }

        if (action < 0) break;
        state.ApplyAction(action);
        taken.push_back(action);
    }

    return taken;
}

// =========================================================================
// ParallelMinimaxBot.
//
// Persistent worker pool + per-worker MinimaxEngine. The root's legal
// actions are partitioned round-robin across workers; each worker scores
// its slice using engine.Search() on the child of the root action (at
// depth-1) and the orchestrator reduces to the global best.
//
// Strict-parity scoring for terminal children mirrors MinimaxEngine::Search
// bit-exactly. Non-terminal children are scored by engine.Search(child,
// depth-1) whose result is from the child's current-player POV; we negate
// when the side-to-move flipped (mill/capture branches don't flip).
// =========================================================================
struct ParallelMinimaxBot::Impl {
    // Workers stay alive for the bot's lifetime; tasks dispatched via cv.
    struct WorkerCtx {
        std::unique_ptr<MinimaxEngine> engine;
        std::thread thread;
        // Inputs for the current dispatch (set by Search, read by worker).
        const State* root_state = nullptr;
        int          depth      = 0;
        // Slice of legal actions to evaluate, addressed via stride/offset:
        // worker i processes legal[i], legal[i+nworkers], legal[i+2*nworkers]...
        // Stored once per dispatch.
        int worker_idx = 0;
        // Output.
        int   best_action = -1;
        int   best_score  = 0;
        long  nodes       = 0;
    };

    bool                          strict      = true;
    int                           num_workers = 1;
    int                           max_score   = 0;  // kWinScore or kRelaxedMate
    std::vector<std::unique_ptr<WorkerCtx>> workers;

    // Dispatch coordination.
    std::mutex                    mu;
    std::condition_variable       cv_start;       // workers wait on this
    std::condition_variable       cv_done;        // orchestrator waits on this
    int                           generation     = 0;  // bumped per Search
    int                           pending_done   = 0;  // remaining workers in current gen
    std::vector<int>              shared_legal;   // current root legal actions
    int                           shared_n       = 0;
    bool                          shutdown       = false;

    void WorkerLoop(WorkerCtx* w) {
        int last_gen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(mu);
            cv_start.wait(lk, [&]{ return shutdown || generation != last_gen; });
            if (shutdown) return;
            last_gen = generation;
            const int my_idx = w->worker_idx;
            const int stride = num_workers;
            const State root = *w->root_state;        // copy under lock for safety
            const int depth  = w->depth;
            std::vector<int> legal = shared_legal;     // copy
            const int n = shared_n;
            lk.unlock();

            // Compute on the copy with no lock held.
            const int cur_player = root.CurrentPlayer();
            int local_best_a = -1;
            int local_best_s = -max_score - 1;
            long local_nodes = 0;

            for (int i = my_idx; i < n; i += stride) {
                const int action = legal[i];
                State child = root;
                child.ApplyAction(action);

                int child_score;
                long search_nodes = 0;
                if (child.IsTerminal()) {
                    const int win = child.Winner();
                    if (win == 2) {
                        child_score = 0;
                    } else if (strict) {
                        child_score = (win == cur_player) ? +kWinScore : -kWinScore;
                        child_score -= (child_score > 0 ?  (100 - depth)
                                                         : -(100 - depth));
                    } else {
                        child_score = (win == cur_player) ?
                                          +(kRelaxedMate - 1) : -(kRelaxedMate - 1);
                    }
                } else if (depth - 1 <= 0) {
                    // Leaf: matches MinimaxSearch / NegamaxStrict's
                    // `depth == 0 -> return Evaluate(s)` bottom-out.
                    // Evaluate is from `child.CurrentPlayer()`'s POV; flip
                    // when the side-to-move switched after the root move.
                    child_score = Evaluate(child);
                    if (child.CurrentPlayer() != cur_player) {
                        child_score = -child_score;
                    }
                    search_nodes = 1;
                } else {
                    SearchResult r = w->engine->Search(child, depth - 1);
                    search_nodes = r.nodes_visited;
                    child_score = r.score;
                    if (child.CurrentPlayer() != cur_player) {
                        child_score = -child_score;
                    }
                }

                local_nodes += search_nodes;
                if (child_score > local_best_s) {
                    local_best_s = child_score;
                    local_best_a = action;
                }
            }

            // Publish result and signal done.
            lk.lock();
            w->best_action = local_best_a;
            w->best_score  = local_best_s;
            w->nodes       = local_nodes;
            if (--pending_done == 0) {
                cv_done.notify_one();
            }
            // Loop back; wait on cv_start for the next dispatch.
        }
    }
};

ParallelMinimaxBot::ParallelMinimaxBot(int num_threads,
                                       std::size_t tt_bytes_per_thread,
                                       bool strict_parity)
    : impl_(new Impl), num_threads_(num_threads < 1 ? 1 : num_threads) {
    impl_->strict      = strict_parity;
    impl_->num_workers = num_threads_;
    impl_->max_score   = strict_parity ? kWinScore : kRelaxedMate;
    impl_->workers.reserve(num_threads_);
    for (int i = 0; i < num_threads_; ++i) {
        auto ctx = std::make_unique<Impl::WorkerCtx>();
        ctx->engine = std::make_unique<MinimaxEngine>(tt_bytes_per_thread,
                                                     strict_parity);
        ctx->worker_idx = i;
        impl_->workers.push_back(std::move(ctx));
    }
    // Spawn worker threads after all WorkerCtx ptrs are stable.
    for (int i = 0; i < num_threads_; ++i) {
        Impl::WorkerCtx* w = impl_->workers[i].get();
        Impl* impl_ptr = impl_;
        w->thread = std::thread([impl_ptr, w]() { impl_ptr->WorkerLoop(w); });
    }
}

ParallelMinimaxBot::~ParallelMinimaxBot() {
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->shutdown = true;
    }
    impl_->cv_start.notify_all();
    for (auto& w : impl_->workers) {
        if (w->thread.joinable()) w->thread.join();
    }
    delete impl_;
}

SearchResult ParallelMinimaxBot::Search(const State& s, int depth) {
    if (s.IsTerminal()) return {-1, 0, 0};
    int buf[kMaxLegalActions];
    const int n = s.LegalActionsInto(buf);
    if (n == 0) {
        return {-1, impl_->strict ? -kWinScore : -kRelaxedMate, 0};
    }
    if (n == 1) {
        // Fast path: one legal move -- no search needed.
        return {buf[0], 0, 0};
    }

    // Fast path for a single worker: use the engine directly to avoid
    // round-tripping through the dispatch cv.
    if (num_threads_ == 1) {
        return impl_->workers[0]->engine->Search(s, depth);
    }

    const int n_active = std::min(num_threads_, n);

    // Publish inputs and wake workers.
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->shared_legal.assign(buf, buf + n);
        impl_->shared_n = n;
        for (auto& w : impl_->workers) {
            w->root_state = &s;
            w->depth      = depth;
            w->best_action = -1;
            w->best_score  = -impl_->max_score - 1;
            w->nodes       = 0;
        }
        impl_->pending_done = num_threads_;
        ++impl_->generation;
    }
    impl_->cv_start.notify_all();

    // Wait for all workers.
    {
        std::unique_lock<std::mutex> lk(impl_->mu);
        impl_->cv_done.wait(lk, [&]{ return impl_->pending_done == 0; });
    }

    // Reduce to global best (workers > n_active just return -1 / sentinel).
    int   best_action = buf[0];
    int   best_score  = -impl_->max_score - 1;
    long  total_nodes = 0;
    for (auto& w : impl_->workers) {
        total_nodes += w->nodes;
        if (w->best_action >= 0 && w->best_score > best_score) {
            best_score  = w->best_score;
            best_action = w->best_action;
        }
    }
    (void)n_active;
    return {best_action, best_score, total_nodes};
}

// =========================================================================
// RunProgressiveEval.
//
// Outer loop: depth = 1, 2, 3, ... while win_rate > 0.5 (or up to max_depth).
// Inner loop: games_per_depth games run concurrently, each on its own
// std::thread. Each thread owns a ParallelMinimaxBot configured with
// threads_per_game = max(1, max_threads / games_per_depth) workers.
//
// Total threads in flight at peak = games_per_depth * threads_per_game
// (clamped <= max_threads). For max_threads=24, games_per_depth=6 we get
// 4 root-split workers per game x 6 games = 24 threads pegged.
//
// The policy_fn is invoked on model turns. Callers in the bindings layer
// re-acquire the GIL inside the std::function wrapper so model forwards
// can run from any worker thread. Multiple workers may call policy_fn
// concurrently; the binding wrapper is responsible for any required
// serialisation (e.g. cuDNN reentrancy).
// =========================================================================
ProgressiveEvalResult RunProgressiveEval(const ProgressiveEvalConfig& cfg,
                                        const ProgressivePolicyFn& policy_fn,
                                        const ProgressiveGameSpecFn& spec_fn,
                                        const ProgressiveProgressFn& progress_fn) {
    ProgressiveEvalResult out;
    if (!policy_fn) {
        throw std::invalid_argument("RunProgressiveEval requires a policy_fn");
    }

    const int games_per_depth = std::max(1, cfg.games_per_depth);
    const int max_threads     = std::max(games_per_depth, cfg.max_threads);
    const int threads_per_game = std::max(1, max_threads / games_per_depth);

    const auto start_wall = std::chrono::steady_clock::now();
    const double budget = cfg.time_budget_s;

    auto effective_spec = [&](int depth, int gi) -> GameSpec {
        if (spec_fn) return spec_fn(depth, gi);
        GameSpec g;
        g.unplaced_white = 9;
        g.unplaced_black = 9;
        g.ai_player      = gi % 2;
        return g;
    };

    int depth = 1;
    while (true) {
        if (!cfg.unlimited && depth > cfg.max_depth) break;
        if (budget > 0.0) {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start_wall).count();
            if (elapsed >= budget) break;
        }

        struct GameOutcome {
            int  result = 0;   // +1 win, 0 draw, -1 loss for the AI
            long nodes  = 0;
        };
        std::vector<GameOutcome> outcomes(games_per_depth);
        std::vector<std::thread> threads;
        threads.reserve(games_per_depth);

        // Exceptions thrown inside policy_fn (e.g. KeyboardInterrupt) must
        // propagate out. Each worker stashes its exception and the outer
        // loop rethrows after join().
        std::vector<std::exception_ptr> errors(games_per_depth);

        const auto depth_start = std::chrono::steady_clock::now();

        for (int gi = 0; gi < games_per_depth; ++gi) {
            const GameSpec spec = effective_spec(depth, gi);
            threads.emplace_back([&, gi, spec]() {
                try {
                    ParallelMinimaxBot bot(threads_per_game,
                                          cfg.tt_bytes_per_thread,
                                          cfg.strict_parity);
                    State s(spec.unplaced_white, spec.unplaced_black);
                    if (s.IsTerminal()) {
                        outcomes[gi] = {0, 0};
                        return;
                    }

                    long nodes = 0;
                    int  steps = 0;
                    while (!s.IsTerminal() && steps < cfg.max_steps) {
                        const int cp = s.CurrentPlayer();
                        int action = -1;
                        if (cp == spec.ai_player) {
                            // Policy callback. The binding is responsible
                            // for re-acquiring the GIL before touching Python.
                            action = policy_fn(s, cp, gi, depth);
                        } else {
                            SearchResult r = bot.Search(s, depth);
                            action = r.action;
                            nodes += r.nodes_visited;
                        }
                        if (action < 0) break;
                        s.ApplyAction(action);
                        ++steps;
                    }

                    int result = 0;
                    if (s.IsTerminal()) {
                        auto r = s.Returns();
                        if (r[spec.ai_player] > r[1 - spec.ai_player]) result = +1;
                        else if (r[spec.ai_player] < r[1 - spec.ai_player]) result = -1;
                        else result = 0;
                    }
                    outcomes[gi] = {result, nodes};
                } catch (...) {
                    errors[gi] = std::current_exception();
                }
            });
        }

        for (auto& t : threads) t.join();
        for (auto& ep : errors) {
            if (ep) std::rethrow_exception(ep);
        }

        DepthResult dr;
        for (const auto& o : outcomes) {
            if (o.result > 0)      ++dr.wins;
            else if (o.result < 0) ++dr.losses;
            else                   ++dr.draws;
            dr.total_nodes += o.nodes;
        }
        dr.win_rate = (dr.wins + 0.5 * dr.draws) /
                      static_cast<double>(games_per_depth);
        dr.wall_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - depth_start).count();
        out.per_depth[depth] = dr;

        if (progress_fn) progress_fn(depth, dr);

        if (dr.win_rate > 0.5) {
            out.max_depth_beaten = depth;
            ++depth;
        } else {
            break;
        }
    }

    out.total_wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_wall).count();
    return out;
}

}  // namespace fastnmm
