import random, pyspiel
game = pyspiel.load_game("nine_mens_morris")

max_id_seen = -1
max_len_seen = 0
for _ in range(2000):
    s = game.new_initial_state()

    while not s.is_terminal():
        la = s.legal_actions()
        print(la)

        exit()
        if la:
            max_id_seen = max(max_id_seen, max(la))
            max_len_seen = max(max_len_seen, len(la))
            s.apply_action(random.choice(la))
        else:
            break

print("max id seen:", max_id_seen, " (theoretical max:", game.num_distinct_actions()-1, ")")
print("max legal list length seen:", max_len_seen)
