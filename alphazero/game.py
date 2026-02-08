#!/usr/bin/python3
#  -*- coding: utf-8 -*-

"""Minimal abstract game interface for AlphaZero-style training.

Design Principles
-----------------
1. **Minimal**: Every method is required by MCTS or self-play; nothing is
   derivable from other methods.
2. **Orthogonal**: Each method has exactly one responsibility.
3. **Stateless**: All methods are pure functions of their arguments (board,
   action, player).  Game instances hold only immutable configuration.
4. **Future-proof**: The interface contains no grid-specific or
   game-specific assumptions.  New board games (Hex, Santorini, …) can be
   added by implementing this interface alone.
"""


class Game:
    """Abstract base class defining the contract every game must satisfy.

    Subclasses must implement all methods that raise ``NotImplementedError``.
    Optional hooks (``log_status``, ``augment_samples``) have sensible
    defaults and may be overridden when needed.

    Removed / merged APIs (relative to earlier revisions)
    -----------------------------------------------------
    * ``action_space_size`` – derivable from board dimensions already present
      in the config (``rows * columns``).  Use config directly.
    * ``next_player(player)`` – only used internally by game implementations
      inside ``next_state`` and ``get_canonical_form``.  Not part of the
      framework contract; kept as a private helper in each game.
    * ``compute_reward`` – was identical across every implementation.  Now a
      concrete default method using the ``DRAW`` sentinel.
    """

    DRAW = 'draw'
    """Sentinel returned by ``is_terminal_state`` to indicate a draw.

    Using a dedicated sentinel (rather than a game-specific constant like
    ``ChessType.EMPTY``) lets the framework distinguish draws from wins
    without knowledge of game-specific player identifiers.
    """

    # ------------------------------------------------------------------
    # Core game-rule methods (must override)
    # ------------------------------------------------------------------

    def get_initial_state(self):
        """Return the starting ``(board, player)`` tuple."""
        raise NotImplementedError()

    def next_state(self, board, action, player):
        """Apply *action* for *player* and return ``(new_board, next_player)``.

        This is the single canonical way to advance the game state.
        """
        raise NotImplementedError()

    def is_terminal_state(self, board, action, player):
        """Determine whether *board* (reached after *action* by *player*) is terminal.

        Returns
        -------
        winner : player id, ``Game.DRAW``, or ``None``
            * The winning player's identifier if someone won.
            * ``Game.DRAW`` if the game ended in a draw.
            * ``None`` if the game is not over.
        """
        raise NotImplementedError()

    def available_actions(self, board):
        """Return a list of legal action indices for the current *board*."""
        raise NotImplementedError()

    def get_canonical_form(self, board, player):
        """Return *board* from the perspective of *player*.

        The neural network always sees the board as if it is Player 1.
        For symmetric two-player games this typically swaps piece colours
        when *player* is Player 2.
        """
        raise NotImplementedError()

    # ------------------------------------------------------------------
    # Concrete defaults (override only when needed)
    # ------------------------------------------------------------------

    def compute_reward(self, winner, player):
        """Compute the scalar reward for *player* given the game *winner*.

        Returns ``1`` for a win, ``-1`` for a loss, and ``0`` for a draw.

        The default implementation relies on ``is_terminal_state`` returning
        ``Game.DRAW`` for draws.  Override only if your game uses a
        non-standard outcome representation.
        """
        if winner == self.DRAW:
            return 0
        return 1 if winner == player else -1

    def log_status(self, board, counts, actions):
        """Optional hook called by MCTS after each simulation for debugging.

        The default implementation is a no-op.  Override to add
        game-specific visualisation or logging.
        """

    def augment_samples(self, samples):
        """Return augmented training samples (e.g. via board symmetries).

        The default implementation returns *samples* unchanged.
        """
        return samples
