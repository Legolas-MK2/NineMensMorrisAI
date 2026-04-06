"""
Nine Men's Morris Game Wrapper
Fixes OpenSpiel bug where moves to position 0 are incorrectly excluded.

Bug: In nine_mens_morris.cc line 197, the condition `np > 0` should be `np >= 0`
This wrapper adds the missing legal moves to position 0 during the moving phase.
"""

import pyspiel

# Position 0's neighbors (positions that can move to/from position 0)
POSITION_0_NEIGHBORS = [1, 9]

# Number of board positions
NUM_POINTS = 24


def _action_from_move(from_pos: int, to_pos: int) -> int:
    """Convert a move (from, to) to an action number."""
    return from_pos * NUM_POINTS + to_pos + NUM_POINTS


def _move_from_action(action: int) -> tuple:
    """Convert an action number to a move (from, to)."""
    if action < NUM_POINTS:
        return None  # Placement action
    adjusted = action - NUM_POINTS
    return (adjusted // NUM_POINTS, adjusted % NUM_POINTS)


class FixedNineMensMorrisState:
    """Wrapper around pyspiel state that fixes the position 0 bug."""

    def __init__(self, state):
        self._state = state

    def __getattr__(self, name):
        """Forward all attribute access to the wrapped state."""
        return getattr(self._state, name)

    def __str__(self):
        return str(self._state)

    def __repr__(self):
        return repr(self._state)

    def _is_moving_phase(self) -> bool:
        """Check if we're in the moving phase (not placing, not flying)."""
        # In placing phase, men_to_deploy > 0
        # In flying phase, a player has <= 3 pieces

        # Check for terminal state first
        if self._state.is_terminal():
            return False

        player = self._state.current_player()
        if player < 0:
            return False  # Invalid player (terminal or chance node)

        state_str = str(self._state)

        # Check if still in placement phase
        if "Men to deploy:" in state_str:
            lines = state_str.split('\n')
            for line in lines:
                if "Men to deploy:" in line:
                    parts = line.split(':')[1].strip().split()
                    if any(int(x) > 0 for x in parts):
                        return False  # Still placing

        # Check piece counts for flying phase
        player_char = 'W' if player == 0 else 'B'
        piece_count = state_str.count(player_char)

        if piece_count <= 3:
            return False  # Flying phase - all moves already legal

        return True  # Moving phase

    def _get_player_positions(self, player: int) -> list:
        """Get list of positions occupied by the given player."""
        state_str = str(self._state)
        positions = []

        # Parse the board - need to map visual positions to indices
        # The board layout maps characters at specific positions to indices 0-23
        # This is a simplified approach - we check if neighbors have pieces

        # Actually, let's check the legal actions to infer piece positions
        # If there's a move from position X, player has a piece at X
        for action in self._state.legal_actions():
            move = _move_from_action(action)
            if move is not None:
                from_pos, _ = move
                if from_pos not in positions:
                    positions.append(from_pos)

        return positions

    def _position_is_empty(self, pos: int) -> bool:
        """Check if a position is empty by examining if any move goes there."""
        # If it's a placement phase, check placement actions
        for action in self._state.legal_actions():
            if action < NUM_POINTS and action == pos:
                return True  # Can place here = empty
            move = _move_from_action(action)
            if move is not None:
                _, to_pos = move
                if to_pos == pos:
                    return True  # Can move here = empty

        # Also check: if position 0 can be moved FROM, it's not empty
        for action in self._state.legal_actions():
            move = _move_from_action(action)
            if move is not None:
                from_pos, _ = move
                if from_pos == pos:
                    return False  # Has our piece

        # Need a more reliable check - parse the board string
        state_str = str(self._state)
        # Position 0 is the top-left corner (first 'W', 'B', or '.' in the board)
        # The board format shows positions as characters

        # The first line is like "W------B------W" where positions are at indices 0, 7, 14
        # Let me just check if there's already a move TO position 0 in legal actions
        # If not, and we're in moving phase, position 0 might be empty but unreachable due to bug

        return True  # Assume empty if we're trying to add a move to it

    def _position_0_is_empty(self) -> bool:
        """Check if position 0 is truly empty by parsing the board string."""
        state_str = str(self._state)
        lines = state_str.split('\n')

        if not lines:
            return False

        # First line of board looks like "W------B------W" or ".------B------W"
        # Position 0 is the very first character
        first_line = lines[0]
        if first_line and first_line[0] == '.':
            return True
        return False

    def legal_actions(self, player=None):
        """Return legal actions with the position 0 bug fixed."""
        original_actions = list(self._state.legal_actions(player) if player is not None
                                else self._state.legal_actions())

        # Only fix in moving phase
        if not self._is_moving_phase():
            return original_actions

        # Check if position 0 is truly empty (no piece there - ours or opponent's)
        if not self._position_0_is_empty():
            return original_actions  # Position 0 occupied, no fix needed

        # Position 0 is empty - check if neighbors have our pieces
        # and if they can move to position 0
        added_actions = []

        for neighbor in POSITION_0_NEIGHBORS:
            # Check if we have a piece at this neighbor (can move FROM there)
            has_piece_at_neighbor = False
            for action in original_actions:
                move = _move_from_action(action)
                if move is not None:
                    from_pos, _ = move
                    if from_pos == neighbor:
                        has_piece_at_neighbor = True
                        break

            if has_piece_at_neighbor:
                # We have a piece at the neighbor, check if move to 0 exists
                move_to_0 = _action_from_move(neighbor, 0)
                if move_to_0 not in original_actions:
                    # This is the bug! Add the missing action
                    added_actions.append(move_to_0)

        if added_actions:
            return sorted(original_actions + added_actions)

        return original_actions

    def clone(self):
        """Clone the state, returning a wrapped copy."""
        return FixedNineMensMorrisState(self._state.clone())

    def child(self, action):
        """Apply action and return wrapped child state."""
        return FixedNineMensMorrisState(self._state.child(action))


class FixedNineMensMorrisGame:
    """Wrapper around pyspiel game that returns fixed states."""

    def __init__(self, game):
        self._game = game

    def __getattr__(self, name):
        """Forward all attribute access to the wrapped game."""
        return getattr(self._game, name)

    def new_initial_state(self):
        """Return a fixed initial state."""
        return FixedNineMensMorrisState(self._game.new_initial_state())


def load_game(name: str = "nine_mens_morris"):
    """Load the Nine Men's Morris game with the position 0 bug fix."""
    if name != "nine_mens_morris":
        return pyspiel.load_game(name)

    game = pyspiel.load_game(name)
    return FixedNineMensMorrisGame(game)


# For drop-in replacement
def load_game_fixed():
    """Convenience function to load the fixed game."""
    return load_game("nine_mens_morris")
