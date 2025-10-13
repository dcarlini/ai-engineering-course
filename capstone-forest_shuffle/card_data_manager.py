import csv
import os

class Card:
    def __init__(self, card_name, type_1, type_2, suit, expansion):
        self.card_name = card_name
        self.type_1 = type_1
        self.type_2 = type_2
        self.suit = suit
        self.expansion = expansion

    def __repr__(self):
        return f"Card(name='{self.card_name}', type_1='{self.type_1}', type_2='{self.type_2}', suit='{self.suit}', expansion='{self.expansion}')"

class CardCollection:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.cards = []
        self._load_cards()

    def _load_cards(self):
        if not os.path.exists(self.csv_file):
            print(f"Error: {self.csv_file} not found.")
            return

        with open(self.csv_file, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                if len(row) == 5:
                    self.cards.append(Card(*row))
                else:
                    print(f"Warning: Skipping malformed row in {self.csv_file}: {row}")

    def search_by_name(self, name):
        return [card for card in self.cards if name.lower() in card.card_name.lower()]

    def search_by_type1(self, type_1):
        return [card for card in self.cards if type_1.lower() == card.type_1.lower()]

    def search_by_type2(self, type_2):
        return [card for card in self.cards if type_2.lower() == card.type_2.lower()]

    def search_by_suit(self, suit):
        return [card for card in self.cards if suit.lower() == card.suit.lower()]

    def search_by_expansion(self, expansion):
        return [card for card in self.cards if expansion.lower() == card.expansion.lower()]

    def get_all_cards(self):
        return self.cards

if __name__ == '__main__':
    # Example Usage:
    cards_csv_path = os.path.join('config', 'cards_by_name.csv')
    card_collection = CardCollection(cards_csv_path)

    print("All Cards:")
    for card in card_collection.get_all_cards():
        print(card)

    print("\nSearching for 'Hare':")
    hare_cards = card_collection.search_by_name('Hare')
    for card in hare_cards:
        print(card)

    print("\nSearching for type_1 'Pawed Animal':")
    pawed_cards = card_collection.search_by_type1('Pawed Animal')
    for card in pawed_cards:
        print(card)

    print("\nSearching for suit 'Linden':")
    linden_cards = card_collection.search_by_suit('Linden')
    for card in linden_cards:
        print(card)

    print("\nSearching for expansion 'Linden':")
    linden_cards = card_collection.search_by_expansion('Linden')
    for card in linden_cards:
        print(card)
