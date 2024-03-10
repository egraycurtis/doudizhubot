import random

def empty_card_dict():
    return {'3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, 'T': 0, 'J': 0, 'Q': 0, 'K': 0, 'A': 0, '2': 0, 'B': 0, 'R': 0}

def empty_card_id_dict():
    return {'3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [], 'T': [], 'J': [], 'Q': [], 'K': [], 'A': [], '2': [], 'B': [], 'R': []}

def rank(card):
    cards = '3456789TJQKA_2BR'
    return cards.index(card)

def mapped_values(dbVal: str):
    if dbVal == '10':
        return 'T'
    if dbVal == 'LJ':
        return 'B'
    if dbVal == 'BJ':
        return 'R'
    return dbVal

def shuffle():
    cards = '3333444455556666777788889999TTTTJJJJQQQQKKKKAAAA2222BR'
    card_list = list(cards)
    random.shuffle(card_list)
    hands = [empty_card_dict(), empty_card_dict(), empty_card_dict()]
    for i in range(len(hands)):
        for card in card_list[i*17:(i+1)*17]:
            hands[i][card] += 1

    middle_cards_taken = False
    for hand in hands[:2]:
        if has_three_trumps(hand):
            print()
            middle_cards_taken = True
            for card in card_list[51:54]:
                hand[card] += 1
            break

    if not middle_cards_taken:
        for card in card_list[51:54]:
            hands[2][card] += 1

    return hands

def has_three_trumps(hand: dict[str, int]):
    trumps = 0
    trumps += hand['B']
    trumps += hand['R']
    if trumps == 2:
        return True
    trumps += hand['2']
    for _, count in hand.items():
        if count == 4:
             trumps += 1
    return trumps >= 3
