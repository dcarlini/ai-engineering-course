import csv
import os

class Symbol:
    def __init__(self, group, filename, display_name):
        self.group = group
        self.filename = filename
        self.display_name = display_name

    def __repr__(self):
        return f"Symbol(group='{self.group}', filename='{self.filename}', display_name='{self.display_name}')"

class SymbolCollection:
    def __init__(self, csv_file='config/symbols.csv'):
        self.csv_file = csv_file
        self.symbols = []
        self._load_symbols()

    def _load_symbols(self):
        if not os.path.exists(self.csv_file):
            print(f"Error: {self.csv_file} not found.")
            return

        with open(self.csv_file, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                if len(row) == 3:
                    self.symbols.append(Symbol(*row))
                else:
                    print(f"Warning: Skipping malformed row in {self.csv_file}: {row}")

    def search_by_display_name(self, display_name):
        return [symbol for symbol in self.symbols if display_name.lower() in symbol.display_name.lower()]

    def search_by_group(self, group):
        return [symbol for symbol in self.symbols if group.lower() == symbol.group.lower()]

    def get_all_symbols(self):
        return self.symbols

if __name__ == '__main__':
    # Example Usage:
    symbol_collection = SymbolCollection()

    print("All Symbols:")
    for symbol in symbol_collection.get_all_symbols():
        print(symbol)

    print("\nSearching for display name 'Tree':")
    tree_symbols = symbol_collection.search_by_display_name('Tree')
    for symbol in tree_symbols:
        print(symbol)

    print("\nSearching for group 'types':")
    type_symbols = symbol_collection.search_by_group('types')
    for symbol in type_symbols:
        print(symbol)
