"""Compact, versioned transport for self-play trajectories.

Workers send only initial hands, selected action ids, and selected values.  The
learner deterministically reconstructs tensors, keeping the queue small without
losing any training information.
"""

from __future__ import annotations

import json
import struct

from cards import empty_card_dict, rank


TRAJECTORY_FORMAT = "trajectory-v2"
_MAGIC = b"DDZ2"
_CARD_ORDER = "3456789TJQKA2BR"
_CARD_ID = {card: index for index, card in enumerate(_CARD_ORDER)}
_HAND_LENGTHS = (20, 17, 17)


def hand_to_string(card_dict: dict[str, int]) -> str:
    return "".join(card * card_dict[card] for card in _CARD_ORDER)


def string_to_hand(cards: str) -> dict[str, int]:
    hand = empty_card_dict()
    for card in cards:
        hand[card] += 1
    return hand


def _pack_hand(cards: str) -> bytes:
    values = [_CARD_ID[card] for card in cards]
    if len(values) % 2:
        values.append(15)
    return bytes((values[index] << 4) | values[index + 1] for index in range(0, len(values), 2))


def _unpack_hand(raw: bytes, length: int) -> str:
    values = []
    for value in raw:
        values.extend((value >> 4, value & 0x0F))
    return "".join(_CARD_ORDER[value] for value in values[:length])


def encode_training_batch(model_name: str, games: list[dict]) -> bytes:
    """Encode a batch without JSON tensor payloads.

    Action ids are unsigned shorts and selected predictions are float16.  The
    latter is sufficient for the existing soft target while shrinking queue I/O.
    """
    name = model_name.encode("utf-8")
    if len(name) > 255 or len(games) > 65535:
        raise ValueError("training batch is too large")
    payload = bytearray(_MAGIC + struct.pack("!BH", len(name), len(games)) + name)
    for game in games:
        hands = game["hands"]
        if len(hands) != 3:
            raise ValueError("each game must contain three hands")
        payload.extend(struct.pack("!B", int(bool(game["landlord_won"]))))
        for hand, length in zip(hands, _HAND_LENGTHS):
            if len(hand) != length:
                raise ValueError("unexpected initial hand length")
            payload.extend(_pack_hand(hand))
        actions = game["actions"]
        predictions = game["predictions"]
        if len(actions) != len(predictions) or len(actions) > 255:
            raise ValueError("invalid trajectory")
        payload.extend(struct.pack("!B", len(actions)))
        for action_id, prediction in zip(actions, predictions):
            payload.extend(struct.pack("!He", int(action_id), float(prediction)))
    return bytes(payload)


def _decode_v2(raw: bytes) -> dict:
    offset = len(_MAGIC)
    name_length, game_count = struct.unpack_from("!BH", raw, offset)
    offset += 3
    model_name = raw[offset : offset + name_length].decode("utf-8")
    offset += name_length
    games = []
    for _ in range(game_count):
        landlord_won = bool(raw[offset])
        offset += 1
        hands = []
        for length in _HAND_LENGTHS:
            byte_length = (length + 1) // 2
            hands.append(_unpack_hand(raw[offset : offset + byte_length], length))
            offset += byte_length
        action_count = raw[offset]
        offset += 1
        actions = []
        predictions = []
        for _ in range(action_count):
            action_id, prediction = struct.unpack_from("!He", raw, offset)
            offset += 4
            actions.append(action_id)
            predictions.append(float(prediction))
        games.append(
            {
                "hands": hands,
                "actions": actions,
                "predictions": predictions,
                "landlord_won": landlord_won,
            }
        )
    if offset != len(raw):
        raise ValueError("trailing bytes in trajectory payload")
    return {"format": TRAJECTORY_FORMAT, "model_name": model_name, "games": games}


def decode_training_batch(raw_payload: bytes | str) -> dict:
    if isinstance(raw_payload, bytes) and raw_payload.startswith(_MAGIC):
        return _decode_v2(raw_payload)
    if isinstance(raw_payload, bytes):
        raw_payload = raw_payload.decode("utf-8")
    return json.loads(raw_payload)
