from cards import rank_with_gap

def are_equal(cards):
    return all(card == cards[0] for card in cards[1:])

def is_straight(cards):
    for i in range(1, len(cards)):
        if rank_with_gap(cards[i]) != rank_with_gap(cards[i-1]) + 1:
            return False
    return True

def is_straight_of_pairs(cards):
    if len(cards) % 2 != 0:
        return False

    zero_elements = cards[::2]
    one_elements = cards[1::2]

    return is_straight(zero_elements) and is_straight(one_elements)

def is_straight_of_triples(cards):
    if len(cards) % 3 != 0:
        return False

    zero_elements = cards[::3]
    one_elements = cards[1::3]
    two_elements = cards[2::3]

    return is_straight(zero_elements) and is_straight(one_elements) and is_straight(two_elements)

def is_straight_of_triples_with_discards(cards):
    if len(cards) % 4 != 0:
        return False

    triples = {}
    for i in range(len(cards) - 2):
        if are_equal(cards[i:i+3]):
            triples[cards[i]] = cards[i]

    cards_with_triples = list(triples.values())
    cards_with_triples.sort(key=lambda card: rank_with_gap(card))

    max_consecutive = 1
    current_consecutive = 1
    for i in range(len(cards_with_triples) - 1):
        if rank_with_gap(cards_with_triples[i]) + 1 == rank_with_gap(cards_with_triples[i + 1]):
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1

    return max_consecutive >= len(cards) / 4

def straight_of_triples_with_discards_rank(cards):
    triples = {}
    for i in range(len(cards) - 2):
        if are_equal(cards[i:i+3]):
            triples[cards[i]] = cards[i]

    cards_with_triples = list(triples.values())
    cards_with_triples.sort(key=lambda card: rank_with_gap(card))

    max_consecutive = 1
    current_consecutive = 1
    for i in range(len(cards_with_triples) - 1):
        if rank_with_gap(cards_with_triples[i]) + 1 == rank_with_gap(cards_with_triples[i + 1]):
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1

    for i in range(len(cards_with_triples) - int(len(cards) / 4) + 1):
        if is_straight(cards_with_triples[i:i + int(len(cards) / 4)]):
            return rank_with_gap(cards_with_triples[i])

    return 0

def is_bomb_with_discards(cards):
    if len(cards) != 6:
        return False

    return are_equal(cards[0:4]) or are_equal(cards[1:5]) or are_equal(cards[2:6])

def bomb_with_discards_rank(cards):
    for i in range(3):
        if are_equal(cards[i:i+4]):
            return rank_with_gap(cards[i])
    return 0


def expected_value(win_probability: float, option_dict: dict[str, int], cards_remaining_dict: dict[str, int]) -> int:
    remaining_cards_multiplier = 1
    if win_probability > .5:
        if cards_remaining_dict['B'] + cards_remaining_dict['R'] == 2:
            remaining_cards_multiplier *= 2
        
        for count in cards_remaining_dict.values():
            if count == 4:
                remaining_cards_multiplier *= 2

    option_multiplier = 1
    has_four_of_a_kind = False
    total_count = 0
    for count in option_dict.values():
        total_count += count
        if count == 4:
            has_four_of_a_kind = True

    if (option_dict['B'] + option_dict['R'] == 2 and total_count == 2 ) or (has_four_of_a_kind and total_count == 4):
        option_multiplier = 2

    return remaining_cards_multiplier * option_multiplier * (win_probability - .5)

def choice_bomb_multiplier(card_dict: dict[str, int]) -> int:
    if card_dict['B'] + card_dict['R'] == 2:
        return 2
    
    has_four_of_a_kind = False
    total_count = 0
    for count in card_dict.values():
        total_count += count
        if count == 4:
            has_four_of_a_kind = True

    if has_four_of_a_kind and total_count == 4:
        return 2

    return 1

def get_turn_info(card_dict: dict[str, int]):
    cards = []
    for card, count in card_dict.items():
        for _ in range(count):
            cards.append(card)
    cards.sort(key=lambda card: rank_with_gap(card))
    info = {'type': 'illegal', 'size': len(cards)}
    if len(cards) == 0:
        info['type'] = 'pass'
    elif len(cards) == 1:
        info['type'] = 'single'
        info['rank'] = rank_with_gap(cards[0])
    elif len(cards) == 2:
        if are_equal(cards):
            info['type'] = 'pair'
            info['rank'] = rank_with_gap(cards[0])
        elif cards[0] == "B" and cards[1] == "R":
            info['type'] = 'bomb'
            info['rank'] = rank_with_gap(cards[0])
    elif len(cards) == 3:
        if are_equal(cards):
            info['type'] = 'triple'
            info['rank'] = rank_with_gap(cards[0])
    elif len(cards) == 4:
        if are_equal(cards):
            info['type'] = 'bomb'
            info['rank'] = rank_with_gap(cards[0])
        elif are_equal(cards[0:3]) or are_equal(cards[1:4]):
            info['type'] = 'triple_with_discard'
            info['rank'] = rank_with_gap(cards[1])
    else:
        if is_straight(cards):
            info['type'] = 'straight'
            info['rank'] = rank_with_gap(cards[0])
        elif is_straight_of_pairs(cards):
            info['type'] = 'straight_of_pairs'
            info['rank'] = rank_with_gap(cards[0])
        elif is_straight_of_triples(cards):
            info['type'] = 'straight_of_triples'
            info['rank'] = rank_with_gap(cards[0])
        elif is_straight_of_triples_with_discards(cards):
            info['type'] = 'straight_of_triples_with_discards'
            info['rank'] = straight_of_triples_with_discards_rank(cards)
        elif is_bomb_with_discards(cards):
            info['type'] = 'bomb_with_discards'
            info['rank'] = bomb_with_discards_rank(cards)

    if info['type'] == 'illegal': print(card_dict)
    return info