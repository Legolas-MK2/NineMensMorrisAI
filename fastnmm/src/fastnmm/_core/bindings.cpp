// pybind11 bindings for the Fast Nine Men's Morris engine.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "nine_mens_morris.hpp"

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

    // ----- SharedMoveCache (POSIX shm, lock-free) -----
    py::class_<SharedMoveCache>(m, "SharedMoveCache",
        "Lock-free best-move cache shared across processes via POSIX "
        "shared memory. Maps Zobrist key -> last best action chosen at "
        "that position. Used by MinimaxEngine for cross-worker move-"
        "ordering hints.")
        .def(py::init<const std::string&, std::size_t, bool>(),
             py::arg("name"),
             py::arg("total_bytes"),
             py::arg("create"),
             "Create or attach a POSIX shm segment named `name` (must "
             "start with '/'). On create, `total_bytes` is rounded down "
             "to a power-of-two count of 16-byte entries. On attach, "
             "the existing size is discovered automatically.")
        .def_static("create",
            [](const std::string& name, std::size_t total_bytes) {
                return new SharedMoveCache(name, total_bytes, true);
            },
            py::arg("name"), py::arg("total_bytes"),
            "Create a new shared cache. Call once from the trainer.")
        .def_static("attach",
            [](const std::string& name) {
                return new SharedMoveCache(name, 0, false);
            },
            py::arg("name"),
            "Attach to an existing shared cache. Call once per worker.")
        .def("get",
             [](const SharedMoveCache& c, uint64_t key) { return c.Get(key); },
             py::arg("key"),
             "Return the cached action for `key`, or -1 on miss.")
        .def("put",
             [](SharedMoveCache& c, uint64_t key, int action) {
                 c.Put(key, action);
             },
             py::arg("key"), py::arg("action"),
             "Store `action` at `key`. No-op if key == 0 or the bucket "
             "is locally dense.")
        .def("close",  &SharedMoveCache::Close,
             "Detach from the shared segment (call in every process).")
        .def("unlink", &SharedMoveCache::Unlink,
             "Destroy the segment (creator-only; call once at shutdown).")
        .def_property_readonly("name",          &SharedMoveCache::Name)
        .def_property_readonly("num_entries",   &SharedMoveCache::NumEntries)
        .def_property_readonly("bytes",         &SharedMoveCache::Bytes)
        .def_property_readonly("probes",        &SharedMoveCache::Probes)
        .def_property_readonly("hits",          &SharedMoveCache::Hits)
        .def_property_readonly("stores",        &SharedMoveCache::Stores)
        .def_property_readonly("store_misses",  &SharedMoveCache::StoreMisses)
        .def_property_readonly("is_creator",    &SharedMoveCache::IsCreator)
        .def("__repr__", [](const SharedMoveCache& c) {
            std::ostringstream os;
            os << "<SharedMoveCache name='" << c.Name()
               << "' entries=" << c.NumEntries()
               << " bytes=" << c.Bytes()
               << " probes=" << c.Probes()
               << " hits=" << c.Hits()
               << " hit_rate=" << (c.Probes() == 0 ? 0.0
                                    : double(c.Hits()) / double(c.Probes()))
               << ">";
            return os.str();
        });

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
        .def("set_root_cache",
             [](MinimaxEngine& e, py::object cache_obj) {
                 if (cache_obj.is_none()) {
                     e.SetRootCache(nullptr);
                 } else {
                     e.SetRootCache(cache_obj.cast<SharedMoveCache*>());
                 }
             },
             py::arg("cache").none(true),
             "Attach a SharedMoveCache to be consulted at the search "
             "root for move-ordering hints and updated with the chosen "
             "root action after each search. Pass None to detach. The "
             "engine does NOT own the cache; the caller must keep it "
             "alive for the engine's lifetime.",
             py::keep_alive<1, 2>())
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
}
