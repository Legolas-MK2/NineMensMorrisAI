// pybind11 bindings for the Fast Nine Men's Morris engine.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "nine_mens_morris.hpp"

#include <exception>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;
using namespace fastnmm;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Fast Nine Men's Morris engine (bitboard, C++).";

    // --- constants ------------------------------------------------------
    m.attr("NUM_POSITIONS")        = kNumPositions;
    m.attr("NUM_PLAYERS")          = kNumPlayers;
    m.attr("PIECES_PER_PLAYER")    = kPiecesPerPlayer;
    m.attr("NUM_DISTINCT_ACTIONS") = kNumDistinctActions;
    m.attr("MOVE_ACTION_OFFSET")   = kMoveActionOffset;
    m.attr("OBSERVATION_PLANES")   = kObservationPlanes;
    m.attr("OBSERVATION_ROWS")     = kObservationRows;
    m.attr("OBSERVATION_COLS")     = kObservationCols;
    m.attr("OBSERVATION_SIZE")     = kObservationSize;
    m.attr("MAX_GAME_LENGTH")      = kMaxGameLength;
    m.attr("DEFAULT_MAX_TURNS")    = kMaxTurnsDefault;

    // --- Phase enum -----------------------------------------------------
    py::enum_<Phase>(m, "Phase")
        .value("PLACING",  Phase::kPlacing)
        .value("MOVING",   Phase::kMoving)
        .value("CAPTURE",  Phase::kCapture)
        .value("TERMINAL", Phase::kTerminal);

    // --- action encoding helpers ---------------------------------------
    m.def("is_move_action", &IsMoveAction);
    m.def("move_from",      &MoveFrom);
    m.def("move_to",        &MoveTo);
    m.def("encode_move",    &EncodeMove);

    // --- State ----------------------------------------------------------
    py::class_<State>(m, "State")
        .def(py::init<>())
        .def(py::init<int, int>(),
             py::arg("unplaced_white"), py::arg("unplaced_black"),
             "Custom-start state: each player begins with the given number "
             "of stones to place (must be in [1, 9]).")
        .def("legal_actions",
             [](const State& s) { return s.LegalActions(); },
             "Return the list of legal actions in the current state.")
        .def("apply_action",
             [](State& s, int a) { s.ApplyAction(a); },
             py::arg("action"))
        .def("is_terminal", &State::IsTerminal)
        .def("is_chance_node", &State::IsChanceNode)
        .def("current_player", &State::CurrentPlayer)
        .def("returns",
             [](const State& s) {
                 auto r = s.Returns();
                 return std::vector<float>{r[0], r[1]};
             })
        .def("rewards",
             [](const State& s) {
                 auto r = s.Rewards();
                 return std::vector<float>{r[0], r[1]};
             })
        .def("history", &State::History)
        .def("move_number", &State::MoveNumber)
        .def("action_to_string", &State::ActionToString,
             py::arg("player"), py::arg("action"))
        .def("__str__", &State::ToString)
        .def("to_string", &State::ToString)
        .def("observation_string", &State::ObservationString,
             py::arg("player") = 0)
        .def("information_state_string",
             [](const State& /*s*/, int /*player*/) { return std::string(""); },
             py::arg("player") = 0)
        .def("observation_tensor",
             [](const State& s, int player) {
                 auto arr = py::array_t<float>(kObservationSize);
                 s.ObservationTensor(player, arr.mutable_data());
                 // Return flat list-of-floats (matches OpenSpiel's Python API).
                 std::vector<float> out(kObservationSize);
                 std::memcpy(out.data(), arr.data(),
                             sizeof(float) * kObservationSize);
                 return out;
             },
             py::arg("player") = 0)
        .def("observation_tensor_numpy",
             [](const State& s, int player) {
                 auto arr = py::array_t<float>({kObservationPlanes,
                                                kObservationRows,
                                                kObservationCols});
                 s.ObservationTensor(player, arr.mutable_data());
                 return arr;
             },
             py::arg("player") = 0,
             "Return observation as a (5,7,7) numpy array (zero-copy path).")
        .def("clone", &State::Clone)
        .def("serialize", &State::Serialize)
        .def("__copy__",      [](const State& s) { return s.Clone(); })
        .def("__deepcopy__",  [](const State& s, py::dict /*memo*/) {
            return s.Clone();
        })
        .def("__repr__",
             [](const State& s) {
                 std::ostringstream os;
                 os << "<fastnmm.State turn=" << s.Turn()
                    << " phase=" << static_cast<int>(s.CurrentPhase())
                    << " current_player=" << s.CurrentPlayer()
                    << ">";
                 return os.str();
             })

        // Extras exposed as properties for convenience / testing.
        .def("board_bitboard", &State::Board, py::arg("player"),
             "Return the 24-bit bitboard for the given player.")
        .def("men_to_deploy",  &State::MenToDeploy, py::arg("player"))
        .def("men_on_board",   &State::MenOnBoard,  py::arg("player"))
        .def("current_phase",  &State::CurrentPhase)
        .def("turn",           &State::Turn)
        .def("winner",         &State::Winner)
        .def("is_action_legal",&State::IsActionLegal, py::arg("action"))
        .def("set_max_turns",  &State::SetMaxTurns,   py::arg("max_turns"))
        .def("max_turns",      &State::MaxTurns);

    // --- Game -----------------------------------------------------------
    py::class_<Game>(m, "Game")
        .def(py::init<>())
        .def_static("num_distinct_actions",    &Game::NumDistinctActions)
        .def_static("num_players",             &Game::NumPlayers)
        .def_static("min_utility",             &Game::MinUtility)
        .def_static("max_utility",             &Game::MaxUtility)
        .def_static("utility_sum",             &Game::UtilitySum)
        .def_static("max_game_length",         &Game::MaxGameLength)
        .def_static("observation_tensor_shape",
                    []() {
                        auto s = Game::ObservationTensorShape();
                        return std::vector<int>{s[0], s[1], s[2]};
                    })
        .def_static("observation_tensor_size", &Game::ObservationTensorSize)
        .def_static("new_initial_state",
                    py::overload_cast<>(&Game::NewInitialState))
        .def_static("new_initial_state",
                    py::overload_cast<int, int>(&Game::NewInitialState),
                    py::arg("unplaced_white"), py::arg("unplaced_black"))
        .def_static("action_to_string",        &Game::ActionToString,
                    py::arg("player"), py::arg("action"))
        .def_static("deserialize_state",       &Game::Deserialize,
                    py::arg("data"));

    // --- free functions / helpers --------------------------------------
    m.def("new_initial_state",
          py::overload_cast<>(&Game::NewInitialState),
          "Create a fresh initial state (9,9 stones).");
    m.def("new_initial_state_with_stones",
          py::overload_cast<int, int>(&Game::NewInitialState),
          py::arg("unplaced_white"), py::arg("unplaced_black"),
          "Create a fresh initial state with custom starting stones per player.");
    m.def("action_to_string",
          [](int player, int action) { return Game::ActionToString(player, action); },
          py::arg("player"), py::arg("action"));
    m.def("deserialize_state", &Game::Deserialize, py::arg("data"));

    // --- Minimax / evaluation ------------------------------------------
    py::class_<SearchResult>(m, "SearchResult")
        .def_readonly("action",        &SearchResult::action)
        .def_readonly("score",         &SearchResult::score)
        .def_readonly("nodes_visited", &SearchResult::nodes_visited)
        .def("__repr__", [](const SearchResult& r) {
            std::ostringstream os;
            os << "<SearchResult action=" << r.action
               << " score=" << r.score
               << " nodes=" << r.nodes_visited << ">";
            return os.str();
        });

    m.def("minimax_search",
          [](const State& s, int depth) {
              SearchResult r;
              {
                  py::gil_scoped_release release;
                  r = MinimaxSearch(s, depth);
              }
              return r;
          },
          py::arg("state"), py::arg("depth") = 4,
          "Run alpha-beta negamax on `state` to the given ply `depth`. "
          "Returns a SearchResult with the best action, its score, and "
          "the number of nodes visited.");

    m.def("minimax_action",
          [](const State& s, int depth) {
              SearchResult r;
              {
                  py::gil_scoped_release release;
                  r = MinimaxSearch(s, depth);
              }
              return r.action;
          },
          py::arg("state"), py::arg("depth") = 4,
          "Shortcut for `minimax_search(state, depth).action`.");

    m.def("evaluate", &Evaluate, py::arg("state"),
          "Static heuristic value of `state` from its current player's "
          "perspective. Does not search.");

    py::class_<EvalBreakdown>(m, "EvalBreakdown",
        "Detailed breakdown of the static evaluation. All `own_*` / "
        "`opp_*` fields are *raw counts* from the current-player and "
        "opponent perspective. The `*_score` fields are signed and sum "
        "to `total` (== Evaluate(state)).")
        .def_readonly("current_player",     &EvalBreakdown::current_player)
        .def_readonly("endgame",            &EvalBreakdown::endgame)
        .def_readonly("own_material",       &EvalBreakdown::own_material)
        .def_readonly("opp_material",       &EvalBreakdown::opp_material)
        .def_readonly("own_mills",          &EvalBreakdown::own_mills)
        .def_readonly("opp_mills",          &EvalBreakdown::opp_mills)
        .def_readonly("own_open_mills",     &EvalBreakdown::own_open_mills)
        .def_readonly("opp_open_mills",     &EvalBreakdown::opp_open_mills)
        .def_readonly("own_running_mills",  &EvalBreakdown::own_running_mills)
        .def_readonly("opp_running_mills",  &EvalBreakdown::opp_running_mills)
        .def_readonly("own_mill_blocks",    &EvalBreakdown::own_mill_blocks)
        .def_readonly("opp_mill_blocks",    &EvalBreakdown::opp_mill_blocks)
        .def_readonly("own_blocked",        &EvalBreakdown::own_blocked)
        .def_readonly("opp_blocked",        &EvalBreakdown::opp_blocked)
        .def_readonly("own_mobility",       &EvalBreakdown::own_mobility)
        .def_readonly("opp_mobility",       &EvalBreakdown::opp_mobility)
        .def_readonly("w_material",         &EvalBreakdown::w_material)
        .def_readonly("w_mill",             &EvalBreakdown::w_mill)
        .def_readonly("w_open_mill",        &EvalBreakdown::w_open_mill)
        .def_readonly("w_running_mill",     &EvalBreakdown::w_running_mill)
        .def_readonly("w_double_mill",      &EvalBreakdown::w_double_mill)
        .def_readonly("w_mill_block",       &EvalBreakdown::w_mill_block)
        .def_readonly("w_blocked",          &EvalBreakdown::w_blocked)
        .def_readonly("w_mobility",         &EvalBreakdown::w_mobility)
        .def_readonly("material_score",     &EvalBreakdown::material_score)
        .def_readonly("mill_score",         &EvalBreakdown::mill_score)
        .def_readonly("open_mill_score",    &EvalBreakdown::open_mill_score)
        .def_readonly("running_mill_score", &EvalBreakdown::running_mill_score)
        .def_readonly("double_mill_score",  &EvalBreakdown::double_mill_score)
        .def_readonly("mill_block_score",   &EvalBreakdown::mill_block_score)
        .def_readonly("blocked_score",      &EvalBreakdown::blocked_score)
        .def_readonly("mobility_score",     &EvalBreakdown::mobility_score)
        .def_readonly("total",              &EvalBreakdown::total);

    m.def("evaluate_breakdown", &EvaluateBreakdown, py::arg("state"),
          "Return the per-component breakdown of the static evaluation.");

    // ----- AI / reward-shaping evaluator (separate from minimax `evaluate`) -----
    py::class_<AIEvalBreakdown>(m, "AIEvalBreakdown",
        "Raw board feature counts for AI / PPO reward shaping. Fields are "
        "from the perspective of the `player` argument passed to "
        "`evaluate_for_ai_breakdown`. Names mirror the dict keys produced "
        "by `utils.extract_state_features` to keep the Python migration "
        "drop-in.")
        .def_readonly("my_pieces",            &AIEvalBreakdown::my_pieces)
        .def_readonly("opp_pieces",           &AIEvalBreakdown::opp_pieces)
        .def_readonly("my_mills",             &AIEvalBreakdown::my_mills)
        .def_readonly("opp_mills",            &AIEvalBreakdown::opp_mills)
        .def_readonly("my_potential_mills",   &AIEvalBreakdown::my_potential_mills)
        .def_readonly("opp_potential_mills",  &AIEvalBreakdown::opp_potential_mills)
        .def_readonly("my_blocked_mills",     &AIEvalBreakdown::my_blocked_mills)
        .def_readonly("opp_blocked_mills",    &AIEvalBreakdown::opp_blocked_mills)
        .def_readonly("my_unblocked_threats", &AIEvalBreakdown::my_unblocked_threats)
        .def_readonly("opp_unblocked_threats",&AIEvalBreakdown::opp_unblocked_threats)
        .def_readonly("my_double_mills",      &AIEvalBreakdown::my_double_mills)
        .def_readonly("opp_double_mills",     &AIEvalBreakdown::opp_double_mills)
        .def_readonly("my_mobility",          &AIEvalBreakdown::my_mobility)
        .def_readonly("opp_mobility",         &AIEvalBreakdown::opp_mobility)
        .def_readonly("player",               &AIEvalBreakdown::player)
        .def_readonly("endgame",              &AIEvalBreakdown::endgame);

    m.def("evaluate_for_ai_breakdown", &EvaluateForAIBreakdown,
          py::arg("state"), py::arg("player"),
          "Raw board feature counts for AI reward shaping, always from "
          "`player`'s perspective. Separate from minimax `evaluate()` "
          "because PPO Φ(s) must be evaluated from the acting player's "
          "view even when the side-to-move has already flipped (see "
          "worker.py post-action shaping call).");

    m.def("evaluate_for_ai", &EvaluateForAI,
          py::arg("state"), py::arg("player"),
          "C++ static AI evaluation score from `player`'s perspective. "
          "Uses a dedicated integer weight set tuned to roughly mirror "
          "the curriculum's default reward_config (×1000), not the "
          "minimax weights. Use this when you want a single scalar "
          "signal without the curriculum's float scaling. For full "
          "curriculum-aware reward shaping, use "
          "`evaluate_for_ai_breakdown` and combine raw counts in Python.");

    // Pure-C++ random rollouts (releases the GIL so callers can parallelise).
    m.def(
        "random_playouts",
        [](int num_games, uint64_t seed, bool return_lengths) {
            std::vector<int> lengths;
            if (return_lengths) lengths.resize(num_games);
            double sum_p0;
            {
                py::gil_scoped_release release;
                sum_p0 = RandomPlayouts(num_games, seed,
                                        return_lengths ? lengths.data() : nullptr);
            }
            return py::make_tuple(sum_p0, lengths);
        },
        py::arg("num_games"),
        py::arg("seed") = 0,
        py::arg("return_lengths") = false,
        "Play `num_games` random games in pure C++ and return "
        "(sum_of_player0_returns, list_of_game_lengths_or_empty).");

    // Zobrist key for a state (also exposed for tests / external caches).
    m.def("zobrist_key", &ZobristKey, py::arg("state"),
          "Return the 64-bit Zobrist key of `state`. Identical states "
          "yield identical keys; any change in board, unplaced counts, "
          "side-to-move, capture phase, or turn flips it.");

    // ----- MinimaxEngine -----
    py::class_<MinimaxEngine>(m, "MinimaxEngine",
        "Persistent transposition-table-backed alpha-beta engine. "
        "Default mode (strict_parity=True) gives bit-exact match with "
        "`minimax_search`. Pass strict_parity=False to use position-"
        "relative mate-distance scoring (kept as an opt-in -- in normal "
        "forward minimax it gives the same hit rate as strict mode).")
        .def(py::init<std::size_t, bool>(),
             py::arg("tt_bytes") = static_cast<std::size_t>(256ULL * 1024 * 1024),
             py::arg("strict_parity") = true,
             "Allocate a transposition table of approximately `tt_bytes` "
             "bytes (rounded down to the largest power-of-two count of "
             "16-byte entries). 256 MiB - 1 GiB is plenty for typical "
             "training; larger sizes only help by reducing collisions, "
             "not by raising the natural transposition hit rate.")
        .def_property_readonly("strict_parity", &MinimaxEngine::StrictParity)
        .def("search",
             [](MinimaxEngine& eng, const State& s, int depth) {
                 SearchResult r;
                 {
                     py::gil_scoped_release release;
                     r = eng.Search(s, depth);
                 }
                 return r;
             },
             py::arg("state"), py::arg("depth") = 4,
             "Run alpha-beta to `depth` ply, returning a SearchResult.")
        .def("step",
             [](MinimaxEngine& eng, const State& s, int depth) {
                 SearchResult r;
                 {
                     py::gil_scoped_release release;
                     r = eng.Search(s, depth);
                 }
                 return r.action;
             },
             py::arg("state"), py::arg("depth") = 4,
             "Shortcut for `engine.search(state, depth).action`.")
        .def("evaluate",
             [](const MinimaxEngine& eng, const State& s) {
                 return eng.Eval(s);
             },
             py::arg("state"),
             "Return the static heuristic value of `state` (no search).")
        .def("new_game", &MinimaxEngine::NewGame,
             "Bump the generation counter so older entries become "
             "preferred for replacement. Call between unrelated games.")
        .def("tt_clear", &MinimaxEngine::TtClear,
             "Wipe the TT and reset all stat counters.")
        .def_property_readonly("tt_num_entries",  &MinimaxEngine::TtNumEntries)
        .def_property_readonly("tt_bytes",        &MinimaxEngine::TtBytes)
        .def_property_readonly("tt_probes",       &MinimaxEngine::TtProbes)
        .def_property_readonly("tt_hits",         &MinimaxEngine::TtHits)
        .def_property_readonly("tt_stores",       &MinimaxEngine::TtStores)
        .def_property_readonly("tt_collisions",   &MinimaxEngine::TtCollisions)
        .def_property_readonly("tt_fill_fraction",&MinimaxEngine::TtFillFraction)
        .def("__repr__", [](const MinimaxEngine& e) {
            std::ostringstream os;
            os << "<MinimaxEngine entries=" << e.TtNumEntries()
               << " bytes=" << e.TtBytes()
               << " probes=" << e.TtProbes()
               << " hits=" << e.TtHits()
               << " hit_rate="
               << (e.TtProbes() == 0 ? 0.0
                                     : static_cast<double>(e.TtHits()) /
                                       static_cast<double>(e.TtProbes()))
               << ">";
            return os.str();
        });

    // ----- OpponentKind / play_until_player -----
    py::enum_<OpponentKind>(m, "OpponentKind")
        .value("RANDOM",  OpponentKind::kRandom)
        .value("MINIMAX", OpponentKind::kMinimax);

    m.def(
        "play_until_player",
        [](State& state,
           int target_player,
           const std::string& opponent_kind,
           int minimax_depth,
           double random_move_prob,
           py::object engine_obj,
           uint64_t rng_seed) {
            OpponentSpec spec;
            const std::string& k = opponent_kind;
            if (k == "random" || k == "RANDOM") {
                spec.kind = OpponentKind::kRandom;
            } else if (k == "minimax" || k == "MINIMAX") {
                spec.kind = OpponentKind::kMinimax;
            } else {
                throw std::invalid_argument(
                    "opponent_kind must be 'random' or 'minimax'");
            }
            spec.minimax_depth = minimax_depth;
            spec.random_move_prob = random_move_prob;

            MinimaxEngine* engine = nullptr;
            if (!engine_obj.is_none()) {
                engine = engine_obj.cast<MinimaxEngine*>();
            }

            uint64_t rng = rng_seed ? rng_seed : 0xDEADBEEFCAFEBABEULL;
            std::vector<int> taken;
            {
                py::gil_scoped_release release;
                taken = PlayUntilPlayer(state, target_player, spec,
                                        engine, rng);
            }
            return py::make_tuple(taken, rng);
        },
        py::arg("state"),
        py::arg("target_player"),
        py::arg("opponent_kind") = "random",
        py::arg("minimax_depth") = 4,
        py::arg("random_move_prob") = 0.0,
        py::arg("engine")           = py::none(),
        py::arg("rng_seed")         = static_cast<uint64_t>(0),
        "Play opponent moves on `state` (in place) until it is "
        "`target_player`'s turn or the state is terminal. Returns "
        "(list_of_actions_taken, new_rng_state). For `opponent_kind='random'` "
        "the engine argument is ignored; for `'minimax'` an optional "
        "`MinimaxEngine` is reused for its TT (otherwise a one-shot search "
        "is run). `random_move_prob` mixes exploration into a minimax "
        "opponent (matches the existing _RandomizedMinimaxBot wrapper).");

    // ----- ParallelMinimaxBot -----
    py::class_<ParallelMinimaxBot>(m, "ParallelMinimaxBot",
        "Root-splitting parallel alpha-beta bot. Holds `num_threads` "
        "persistent MinimaxEngine instances + a worker pool; each "
        "`search()` call partitions the root's legal actions across "
        "workers and reduces to the global best (action, score). "
        "Bit-exact match with single-threaded MinimaxEngine.search() at "
        "terminal leaves; non-terminal leaves use the engine's "
        "strict_parity probe behaviour. NOT safe to call .search() "
        "concurrently from multiple Python threads on the same instance "
        "-- construct one per game.")
        .def(py::init<int, std::size_t, bool>(),
             py::arg("num_threads") = 4,
             py::arg("tt_bytes_per_thread") = static_cast<std::size_t>(64ULL * 1024 * 1024),
             py::arg("strict_parity") = true,
             "Allocate `num_threads` workers, each with its own "
             "MinimaxEngine of `tt_bytes_per_thread` bytes.")
        .def("search",
             [](ParallelMinimaxBot& bot, const State& s, int depth) {
                 SearchResult r;
                 {
                     py::gil_scoped_release release;
                     r = bot.Search(s, depth);
                 }
                 return r;
             },
             py::arg("state"), py::arg("depth") = 4,
             "Root-split alpha-beta search to `depth` ply. Returns a "
             "SearchResult.")
        .def("step",
             [](ParallelMinimaxBot& bot, const State& s, int depth) {
                 SearchResult r;
                 {
                     py::gil_scoped_release release;
                     r = bot.Search(s, depth);
                 }
                 return r.action;
             },
             py::arg("state"), py::arg("depth") = 4,
             "Shortcut for `bot.search(state, depth).action`.")
        .def_property_readonly("num_threads", &ParallelMinimaxBot::NumThreads);

    // ----- ProgressiveEval -----
    py::class_<GameSpec>(m, "ProgressiveGameSpec",
        "Per-game configuration produced by the `spec_fn` callback in "
        "`progressive_minimax_eval`. Controls stone counts and which "
        "side the AI plays in a single game.")
        .def(py::init<>())
        .def_readwrite("unplaced_white", &GameSpec::unplaced_white)
        .def_readwrite("unplaced_black", &GameSpec::unplaced_black)
        .def_readwrite("ai_player",      &GameSpec::ai_player)
        .def("__repr__", [](const GameSpec& g) {
            std::ostringstream os;
            os << "<ProgressiveGameSpec uw=" << g.unplaced_white
               << " ub=" << g.unplaced_black
               << " ai_player=" << g.ai_player << ">";
            return os.str();
        });

    py::class_<DepthResult>(m, "ProgressiveDepthResult",
        "Per-depth result block produced by `progressive_minimax_eval`. "
        "Matches the dict shape returned by the existing Python "
        "`evaluate_vs_minimax` (`wins`, `draws`, `losses`, `win_rate`, "
        "`total_nodes`) plus the wall-clock time spent at this depth.")
        .def_readonly("wins",         &DepthResult::wins)
        .def_readonly("draws",        &DepthResult::draws)
        .def_readonly("losses",       &DepthResult::losses)
        .def_readonly("total_nodes",  &DepthResult::total_nodes)
        .def_readonly("win_rate",     &DepthResult::win_rate)
        .def_readonly("wall_seconds", &DepthResult::wall_seconds)
        .def("as_dict", [](const DepthResult& dr) {
            py::dict d;
            d["wins"]         = dr.wins;
            d["draws"]        = dr.draws;
            d["losses"]       = dr.losses;
            d["total_nodes"]  = dr.total_nodes;
            d["win_rate"]     = dr.win_rate;
            d["wall_seconds"] = dr.wall_seconds;
            return d;
        });

    // progressive_minimax_eval(...) -- the top-level entry point.
    //
    // policy_fn: callable(state, current_player, game_idx, depth) -> int
    //   Returns a legal action. Called from C++ worker threads with the
    //   GIL re-acquired. Multiple workers may call it concurrently --
    //   the Python implementation is responsible for any serialisation
    //   (e.g. a model-forward lock for cuDNN reentrancy).
    //
    // spec_fn: callable(depth, game_idx) -> ProgressiveGameSpec | tuple | None
    //   Optional. When provided, decides the AI side and stone counts
    //   for each game. The callback can return either a
    //   ProgressiveGameSpec instance OR a tuple (unplaced_white,
    //   unplaced_black, ai_player). Returning None falls back to the
    //   default (9,9,gi%2).
    //
    // progress_fn: callable(depth, ProgressiveDepthResult) -> None
    //   Optional. Fired after each depth's W/D/L is known. Useful for
    //   streaming progress lines to a logger.
    m.def(
        "progressive_minimax_eval",
        [](py::function policy_fn,
           int max_depth,
           int games_per_depth,
           int max_threads,
           int max_steps,
           std::size_t tt_bytes_per_thread,
           bool unlimited,
           bool strict_parity,
           double time_budget_s,
           py::object spec_fn_obj,
           py::object progress_fn_obj) {

            ProgressiveEvalConfig cfg;
            cfg.max_depth           = max_depth;
            cfg.games_per_depth     = games_per_depth;
            cfg.max_threads         = max_threads;
            cfg.max_steps           = max_steps;
            cfg.tt_bytes_per_thread = tt_bytes_per_thread;
            cfg.unlimited           = unlimited;
            cfg.strict_parity       = strict_parity;
            cfg.time_budget_s       = time_budget_s;

            // Stash any exception raised inside a Python callback so we
            // can re-raise it from the GIL-held outer scope. py::error_already_set
            // captures Python exceptions; we hold it as a generic
            // std::exception_ptr to keep the worker code provider-agnostic.
            std::mutex   err_mu;
            std::exception_ptr stashed;

            // Workers call policy_fn from arbitrary threads. We
            // re-acquire the GIL before touching the Python object.
            auto policy_cb =
                [&](const State& s, int cp, int gi, int d) -> int {
                    py::gil_scoped_acquire gil;
                    try {
                        py::object r = policy_fn(s, cp, gi, d);
                        return r.cast<int>();
                    } catch (py::error_already_set& e) {
                        {
                            std::lock_guard<std::mutex> lk(err_mu);
                            if (!stashed) stashed = std::current_exception();
                        }
                        // Returning -1 short-circuits the inner game loop;
                        // the outer code will propagate the stashed
                        // exception after join().
                        return -1;
                    }
                };

            ProgressiveGameSpecFn spec_cb;
            if (!spec_fn_obj.is_none()) {
                py::function spec_fn = py::reinterpret_borrow<py::function>(spec_fn_obj);
                spec_cb = [&, spec_fn](int d, int gi) -> GameSpec {
                    py::gil_scoped_acquire gil;
                    try {
                        py::object r = spec_fn(d, gi);
                        if (r.is_none()) {
                            GameSpec g;
                            g.ai_player = gi % 2;
                            return g;
                        }
                        try {
                            return r.cast<GameSpec>();
                        } catch (py::cast_error&) {
                            // Fall through to tuple form.
                        }
                        auto t = r.cast<std::tuple<int, int, int>>();
                        GameSpec g;
                        g.unplaced_white = std::get<0>(t);
                        g.unplaced_black = std::get<1>(t);
                        g.ai_player      = std::get<2>(t);
                        return g;
                    } catch (py::error_already_set&) {
                        {
                            std::lock_guard<std::mutex> lk(err_mu);
                            if (!stashed) stashed = std::current_exception();
                        }
                        GameSpec g;
                        g.ai_player = gi % 2;
                        return g;
                    }
                };
            }

            ProgressiveProgressFn progress_cb;
            if (!progress_fn_obj.is_none()) {
                py::function progress_fn = py::reinterpret_borrow<py::function>(progress_fn_obj);
                progress_cb = [&, progress_fn](int d, const DepthResult& dr) {
                    py::gil_scoped_acquire gil;
                    try {
                        progress_fn(d, dr);
                    } catch (py::error_already_set&) {
                        std::lock_guard<std::mutex> lk(err_mu);
                        if (!stashed) stashed = std::current_exception();
                    }
                };
            }

            ProgressiveEvalResult out;
            {
                py::gil_scoped_release release;
                out = RunProgressiveEval(cfg, policy_cb, spec_cb, progress_cb);
            }

            // Re-raise any Python exception caught inside a callback.
            if (stashed) std::rethrow_exception(stashed);

            // Build the Python-facing result dict.
            py::dict per_depth;
            for (const auto& kv : out.per_depth) {
                py::dict d;
                d["wins"]         = kv.second.wins;
                d["draws"]        = kv.second.draws;
                d["losses"]       = kv.second.losses;
                d["total_nodes"]  = kv.second.total_nodes;
                d["win_rate"]     = kv.second.win_rate;
                d["wall_seconds"] = kv.second.wall_seconds;
                per_depth[py::int_(kv.first)] = d;
            }
            py::dict result;
            result["max_depth_beaten"]   = out.max_depth_beaten;
            result["per_depth"]          = per_depth;
            result["total_wall_seconds"] = out.total_wall_seconds;
            return result;
        },
        py::arg("policy_fn"),
        py::arg("max_depth")           = 100,
        py::arg("games_per_depth")     = 6,
        py::arg("max_threads")         = 24,
        py::arg("max_steps")           = 200,
        py::arg("tt_bytes_per_thread") = static_cast<std::size_t>(64ULL * 1024 * 1024),
        py::arg("unlimited")           = true,
        py::arg("strict_parity")       = true,
        py::arg("time_budget_s")       = -1.0,
        py::arg("spec_fn")             = py::none(),
        py::arg("progress_fn")         = py::none(),
        "Run the whole progressive model-vs-minimax climb in C++. \n\n"
        "At each depth, `games_per_depth` games run concurrently on "
        "their own threads; each game's bot uses root-splitting across "
        "`max(1, max_threads // games_per_depth)` worker threads, "
        "saturating the configured core budget during the bot's search "
        "(which dominates wall time). \n\n"
        "policy_fn(state, current_player, game_idx, depth) -> int is "
        "called from the worker threads (GIL re-acquired) on AI turns "
        "and must return a legal action. Multiple workers may call it "
        "concurrently -- the Python side is responsible for serialising "
        "any non-reentrant GPU work (e.g. a single lock around the "
        "model forward). \n\n"
        "Returns a dict with keys: 'max_depth_beaten' (int), 'per_depth' "
        "(dict {depth: {wins, draws, losses, win_rate, total_nodes, "
        "wall_seconds}}), 'total_wall_seconds' (float).");
}
