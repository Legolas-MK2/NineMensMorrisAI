// Fast Nine Men's Morris engine -- implementation.

#include "nine_mens_morris.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <system_error>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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
//   material      (10 * diff)  -- on-board + unplaced pieces
//   mills         ( 3 * diff)  -- completed 3-in-a-row lines
//   mobility      ( 1 * diff)  -- number of legal moves (approx)
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
}  // namespace

int Evaluate(const State& s) {
    if (s.IsTerminal()) {
        const int w = s.Winner();
        if (w == 2) return 0;  // draw
        // Evaluate from perspective of the player "to move" even when
        // terminal (no one is to move, but we still return a signed score).
        // After terminal the current_player is TerminalPlayer; fall back
        // to last mover being the OPPONENT of the one who would move next.
        // Simplest: the winner gets +kWinScore regardless of current_player.
        // Callers only use Evaluate at the root / internal nodes via
        // negamax, which passes them before a node becomes terminal.
        return kWinScore;  // unused from search; see MinimaxSearch.
    }
    const int cp  = s.CurrentPlayer();
    const int opp = 1 - cp;

    const int own_material = s.MenOnBoard(cp)  + s.MenToDeploy(cp);
    const int opp_material = s.MenOnBoard(opp) + s.MenToDeploy(opp);

    const int own_mills = MillCount(s, cp);
    const int opp_mills = MillCount(s, opp);

    const int own_mob = ApproxMobility(s, cp);
    const int opp_mob = ApproxMobility(s, opp);

    return 10 * (own_material - opp_material)
         +  3 * (own_mills    - opp_mills)
         +  1 * (own_mob      - opp_mob);
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
        const int w = s.Winner();
        if (w == 2) return 0;
        // The player to move *would have been* some player; but terminal
        // means the game has ended. To give the loser a negative score,
        // we check the last player to move. Simpler: rely on EndGame having
        // set Returns properly -- +1 to winner, -1 to loser. Since we need
        // a score from the perspective of the state's "to-move" player,
        // and there's no to-move player, we use the sign convention from
        // the *negamax caller's* perspective. The caller recurses with the
        // convention `score = -Negamax(child)`, so by the time we hit a
        // terminal, the caller expects the score from THEIR perspective.
        // We use the fact that the last move was made by the OPPOSITE of
        // the frozen `current_player_`. In our engine, on terminal we set
        // current_player_ to kTerminalPlayer so we can't rely on it.
        //
        // Workaround: encode terminal score using s.Winner() and the
        // depth-adjusted magnitude, then let the caller negate as usual.
        // Convention: returns are "for player 0". Transform to "for the
        // caller's parent's current player":
        //
        // In practice: the simplest robust thing is to ensure we never
        // recurse into terminal nodes without immediately resolving them
        // with the right sign. See the caller: after ApplyAction, if the
        // child is terminal, we evaluate in the child's frame.
        //
        // Here: we return +kWinScore if Returns()[0] > 0, -kWinScore if
        // < 0 -- but we still need the caller's perspective. Let's use
        // a marker: +kWinScore means "the side that just moved won".
        // That's what the caller expects after negation.
        if (w == 0) return +kWinScore;  // White won
        return +kWinScore;              // Black won -- also "mover won"
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
//   - strict (parity tests): bit-exact match to MinimaxSearch. TT probe
//     requires cached_depth == requested_depth.
//   - relaxed (default, training): mate-distance scoring (chess-engine
//     convention) + cached_depth >= requested_depth probe. Massive
//     cross-search hit rates; non-mate move selection identical to
//     strict; mate-distance tie-breaking can differ.
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
    if (depth == 0) return Evaluate(s);

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
    if (depth == 0) return Evaluate(s);

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

    // Optional shared-cache hint: try this move first if it's legal.
    // Bit-exact parity with MinimaxSearch is preserved because we still
    // iterate every legal move below in the engine's natural order; the
    // hint only changes which one is *evaluated first*, not which one is
    // returned (alpha-beta is sound; root has no cutoff).
    int hint_move = -1;
    if (root_cache_ != nullptr) {
        const int candidate = root_cache_->Get(root_key);
        if (candidate >= 0) {
            for (int i = 0; i < n; ++i) {
                if (buf[i] == candidate) { hint_move = candidate; break; }
            }
        }
    }
    if (hint_move >= 0) {
        State child = s;
        child.ApplyAction(hint_move);
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
        best_score  = child_score;
        best_action = hint_move;
        if (best_score > alpha) alpha = best_score;
    }

    for (int i = 0; i < n; ++i) {
        if (buf[i] == hint_move) continue;  // already evaluated above
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
    if (root_cache_ != nullptr && best_action >= 0) {
        root_cache_->Put(root_key, best_action);
    }
    return {best_action, best_score, nodes};
}

int MinimaxEngine::Eval(const State& s) const {
    return Evaluate(s);
}

// =========================================================================
// SharedMoveCache.
// =========================================================================
namespace {

std::size_t RoundDownToPow2(std::size_t n) {
    if (n < 2) return 1;
    std::size_t p = 1;
    while ((p << 1) <= n) p <<= 1;
    return p;
}

}  // namespace

SharedMoveCache::SharedMoveCache(const std::string& name,
                                 std::size_t total_bytes,
                                 bool create) : name_(name) {
    if (name.empty() || name[0] != '/') {
        throw std::invalid_argument(
            "SharedMoveCache: name must start with '/' (POSIX shm convention)");
    }

    if (create) {
        // Round entry count down to power-of-2 so we can use mask indexing.
        std::size_t entries = total_bytes / kBytesPerEntry;
        if (entries < 1024) entries = 1024;
        entries = RoundDownToPow2(entries);
        std::size_t bytes = entries * kBytesPerEntry;

        // Try to create exclusively; if a stale segment from a crashed
        // run exists, unlink it and retry.
        int fd = ::shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
        if (fd == -1 && errno == EEXIST) {
            ::shm_unlink(name.c_str());
            fd = ::shm_open(name.c_str(), O_RDWR | O_CREAT | O_EXCL, 0600);
        }
        if (fd == -1) {
            throw std::system_error(errno, std::generic_category(),
                "SharedMoveCache: shm_open(create) failed for " + name);
        }
        if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
            const int e = errno;
            ::close(fd);
            ::shm_unlink(name.c_str());
            throw std::system_error(e, std::generic_category(),
                "SharedMoveCache: ftruncate failed (try a smaller "
                "total_bytes; /dev/shm may be undersized)");
        }
        void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) {
            const int e = errno;
            ::close(fd);
            ::shm_unlink(name.c_str());
            throw std::system_error(e, std::generic_category(),
                "SharedMoveCache: mmap failed");
        }
        // Linux zero-fills new pages on demand; no explicit memset needed.
        table_       = reinterpret_cast<Entry*>(p);
        num_entries_ = entries;
        mask_        = static_cast<uint64_t>(entries - 1);
        bytes_       = bytes;
        shm_fd_      = fd;
        is_creator_  = true;
    } else {
        // Attach: discover the actual size via fstat.
        int fd = ::shm_open(name.c_str(), O_RDWR, 0600);
        if (fd == -1) {
            throw std::system_error(errno, std::generic_category(),
                "SharedMoveCache: shm_open(attach) failed for " + name);
        }
        struct stat st{};
        if (::fstat(fd, &st) != 0) {
            const int e = errno;
            ::close(fd);
            throw std::system_error(e, std::generic_category(),
                "SharedMoveCache: fstat failed");
        }
        std::size_t bytes = static_cast<std::size_t>(st.st_size);
        std::size_t entries = bytes / kBytesPerEntry;
        if (entries < 2 || (entries & (entries - 1)) != 0) {
            ::close(fd);
            throw std::runtime_error(
                "SharedMoveCache: existing segment has non-power-of-two size");
        }
        void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) {
            const int e = errno;
            ::close(fd);
            throw std::system_error(e, std::generic_category(),
                "SharedMoveCache: mmap failed (attach)");
        }
        table_       = reinterpret_cast<Entry*>(p);
        num_entries_ = entries;
        mask_        = static_cast<uint64_t>(entries - 1);
        bytes_       = bytes;
        shm_fd_      = fd;
        is_creator_  = false;
        (void)total_bytes;  // ignored on attach
    }
}

SharedMoveCache::~SharedMoveCache() {
    Close();
}

void SharedMoveCache::Close() {
    if (closed_) return;
    if (table_ != nullptr && bytes_ > 0) {
        ::munmap(table_, bytes_);
        table_ = nullptr;
    }
    if (shm_fd_ != -1) {
        ::close(shm_fd_);
        shm_fd_ = -1;
    }
    closed_ = true;
}

void SharedMoveCache::Unlink() {
    if (!is_creator_) return;
    ::shm_unlink(name_.c_str());
}

int SharedMoveCache::Get(uint64_t key) const {
    ++probes_;
    if (key == 0) return kNoMove;  // 0 reserved as empty sentinel
    std::size_t idx = static_cast<std::size_t>(key & mask_);
    for (int i = 0; i < kProbeMax; ++i) {
        const uint64_t k = table_[idx].key.load(std::memory_order_relaxed);
        if (k == key) {
            const int32_t a = table_[idx].action.load(std::memory_order_relaxed);
            ++hits_;
            return static_cast<int>(a);
        }
        if (k == 0) return kNoMove;          // empty -> definite miss
        idx = (idx + 1) & mask_;
    }
    return kNoMove;
}

void SharedMoveCache::Put(uint64_t key, int action) {
    if (key == 0) return;
    std::size_t idx = static_cast<std::size_t>(key & mask_);
    for (int i = 0; i < kProbeMax; ++i) {
        const uint64_t k = table_[idx].key.load(std::memory_order_relaxed);
        if (k == 0 || k == key) {
            // Write action BEFORE key. A torn reader that observes the
            // new key with the old action gets a stale move (still
            // legal-validated by the consumer); a reader that observes
            // the old key with the new action gets a key mismatch and
            // falls through.
            table_[idx].action.store(static_cast<int32_t>(action),
                                     std::memory_order_relaxed);
            table_[idx].key.store(key, std::memory_order_relaxed);
            ++stores_;
            return;
        }
        idx = (idx + 1) & mask_;
    }
    ++store_misses_;  // bucket locally dense; skip
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

}  // namespace fastnmm
