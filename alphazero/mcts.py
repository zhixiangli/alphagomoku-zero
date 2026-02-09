#!/usr/bin/python3
#  -*- coding: utf-8 -*-

import numpy


class MCTS:

    def __init__(self, nnet, game, args):
        self.nnet = nnet
        self.game = game
        self.args = args
        try:
            self._batch_size = args.mcts_batch_size
        except (AttributeError, KeyError):
            self._batch_size = 8

        self.visit_count = {}  # N(s, a) is the visit count
        self.mean_action_value = {}  # Q(s, a) is the mean action value
        self.prior_probability = {}  # P(s, a) is the prior probability of selecting that edge.

        self.terminal_state = {}
        self.total_visit_count = {}
        self.available_actions = {}

    def simulate(self, board, player):
        sim = 0
        while sim < self.args.simulation_num:
            batch = min(self._batch_size, self.args.simulation_num - sim)
            leaves = []
            terminals = []

            # Phase 1: Traverse tree batch times, collecting leaves with virtual losses
            for _ in range(batch):
                path = []
                b, p = board, player
                is_terminal = False

                while b in self.prior_probability:
                    index = self.__select(b)
                    path.append((b, index))
                    # Apply virtual loss to encourage exploration in the batch
                    self.visit_count[b][index] += 1
                    self.total_visit_count[b] += 1

                    action = self.available_actions[b][index]
                    nb, np_ = self.game.next_state(b, action, p)

                    if nb not in self.terminal_state:
                        self.terminal_state[nb] = self.game.is_terminal_state(nb, action, p)
                    if self.terminal_state[nb] is not None:
                        value = 1 if p == self.terminal_state[nb] else 0
                        terminals.append((path, value))
                        is_terminal = True
                        break
                    b, p = nb, np_

                if not is_terminal:
                    leaves.append((path, b, p))
                sim += 1

            # Phase 2: Remove all virtual losses
            for path, _, _ in leaves:
                for b, idx in path:
                    self.visit_count[b][idx] -= 1
                    self.total_visit_count[b] -= 1
            for path, _ in terminals:
                for b, idx in path:
                    self.visit_count[b][idx] -= 1
                    self.total_visit_count[b] -= 1

            # Phase 3: Batch expand unique leaves
            if leaves:
                self.__batch_expand(leaves)

            # Phase 4: Backpropagate terminal results
            for path, term_val in terminals:
                self.__backpropagate(path, -term_val)

        self.game.log_status(board, numpy.copy(self.visit_count[board]), numpy.copy(self.available_actions[board]))
        return numpy.copy(self.available_actions[board]), numpy.copy(self.visit_count[board])

    def search(self, board, player):
        if board not in self.prior_probability:  # leaf
            return -self.__expand(board, player)
        index = self.__select(board)
        action = self.available_actions[board][index]
        next_board, next_player = self.game.next_state(board, action, player)
        if next_board not in self.terminal_state:
            self.terminal_state[next_board] = self.game.is_terminal_state(next_board, action, player)
        if self.terminal_state[next_board] is not None:
            value = 1 if player == self.terminal_state[next_board] else 0
        else:
            value = self.search(next_board, next_player)
        self.__backup(board, index, value)
        return -value

    def __select(self, board):
        u = self.args.c_puct * self.prior_probability[board] * numpy.sqrt(
            self.total_visit_count[board]) / (1.0 + self.visit_count[board])
        values = self.mean_action_value[board] + u
        return int(numpy.argmax(values))

    def __backup(self, board, index, value):
        mav = self.mean_action_value[board]
        vc = self.visit_count[board]
        vc_i = vc[index]
        mav[index] = (mav[index] * vc_i + value) / (vc_i + 1.0)
        vc[index] = vc_i + 1
        self.total_visit_count[board] += 1

    def __expand(self, board, player):
        canonical_board = self.game.get_canonical_form(board, player)
        proba, value = self.nnet.predict(canonical_board)
        actions = self.game.available_actions(board)
        self.available_actions[board] = actions
        self.prior_probability[board] = proba[actions] / numpy.sum(proba[actions])
        self.total_visit_count[board] = 1
        self.mean_action_value[board] = numpy.zeros(len(actions))
        self.visit_count[board] = numpy.zeros(len(actions))
        return value

    def __batch_expand(self, leaves):
        """Expand multiple leaf nodes with a single batched NN prediction."""
        # Deduplicate: if multiple paths lead to same leaf, expand once
        unique = {}  # leaf_board -> index in leaves list
        canonical_forms = []
        for i, (path, leaf_board, leaf_player) in enumerate(leaves):
            if leaf_board not in self.prior_probability and leaf_board not in unique:
                unique[leaf_board] = len(canonical_forms)
                canonical_forms.append(self.game.get_canonical_form(leaf_board, leaf_player))

        if canonical_forms:
            batch = numpy.array(canonical_forms)
            probas, values = self.nnet.batch_predict(batch)

            # Expand each unique leaf
            leaf_values = {}  # leaf_board -> expand_value
            board_list = list(unique.keys())
            for j, leaf_board in enumerate(board_list):
                proba = probas[j]
                actions = self.game.available_actions(leaf_board)
                self.available_actions[leaf_board] = actions
                proba_actions = proba[actions]
                self.prior_probability[leaf_board] = proba_actions / numpy.sum(proba_actions)
                self.total_visit_count[leaf_board] = 1
                self.mean_action_value[leaf_board] = numpy.zeros(len(actions))
                self.visit_count[leaf_board] = numpy.zeros(len(actions))
                leaf_values[leaf_board] = values[j]

        # Backpropagate all leaves
        for path, leaf_board, leaf_player in leaves:
            if leaf_board in leaf_values:
                self.__backpropagate(path, leaf_values[leaf_board])
            elif leaf_board not in self.prior_probability:
                # Leaf not yet expanded (shouldn't normally happen)
                value = self.__expand(leaf_board, leaf_player)
                self.__backpropagate(path, value)
            else:
                # Duplicate leaf already expanded by earlier batch; search subtree
                search_result = self.search(leaf_board, leaf_player)
                self.__backpropagate(path, -search_result)

    def __backpropagate(self, path, init_value):
        """Backpropagate a value through the search path.

        init_value semantics match __expand return: value from the leaf
        player's perspective. The first parent gets -init_value (opponent's
        perspective), alternating at each level.
        """
        value = init_value
        for b, idx in reversed(path):
            value = -value
            mav = self.mean_action_value[b]
            vc = self.visit_count[b]
            vc_i = vc[idx]
            mav[idx] = (mav[idx] * vc_i + value) / (vc_i + 1.0)
            vc[idx] = vc_i + 1
            self.total_visit_count[b] += 1
