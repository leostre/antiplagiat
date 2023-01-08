import ast
import pickle
import re
import argparse

from sklearn.linear_model import LogisticRegression

def main(namespace):

    pairs = Comparer.read_pairs(namespace.input_file)
    with open(namespace.output_file, 'w') as out:
        for pair in pairs:
            cmp = Comparer(*pair, model=namespace.model)
            print(cmp.estimate(), file=out)

def create_parser():
    parser = argparse.ArgumentParser('antiplagiat comparer')
    parser.add_argument('input_file', type=str, help='Path to input file containing paths pairs to compare')
    parser.add_argument('output_file', type=str, help='Path to output file where te scores for each pair are written')
    parser.add_argument('--model', type=str, help='Path to trained model')
    return parser



def keys_only(func):
    def set_wrap(*args, **kwargs):
        wrapped_args = []
        for arg in args:
            if isinstance(arg, dict):
                wrapped_args.append(set(arg))
            else:
                wrapped_args.append(arg)
        return func(*tuple(wrapped_args), **kwargs)

    return set_wrap


class Comparer:
    SIMILARITY_THRESHHOLD = 0.67
    TYPES_OF_INTEREST = ('id', 'name', 'func')
    MODEL = 'model.pkl'

    def __init__(self, path1, path2, model=MODEL):
        self.path1 = path1
        self.path2 = path2
        self.evth1 = Comparer.get_everything(path1)
        self.evth2 = Comparer.get_everything(path2)
        self.model = Comparer.load_model(model)

    def overlapping_by_type(self, type):
        return self.__names_overlapping(self.evth1[type], self.evth2[type])

    def suspicious_by_type(self, type):
        return self.__suspiciuos_names(self.evth1[type], self.evth2[type])

    @keys_only
    def __names_overlapping(self, s1, s2):
        return 2 * len(s1 & s2) / (len(s1) + len(s2))

    @keys_only
    def __suspiciuos_names(self, s1, s2, similarity_threshhold=SIMILARITY_THRESHHOLD):
        mean_size = (len(s1) + len(s2)) / 2

        common = s1 & s2
        s1, s2 = s1 - common, s2 - common
        suspicious = 0
        for i in s1:
            for j in s2:
                if 1 - 2 * Comparer.levenstein_distance(i, j) / (len(i) + len(j)) > similarity_threshhold:
                    suspicious += 1
        return suspicious / mean_size

    @classmethod
    def get_everything(cls, path):
        try:
            walker = TreeWalker(
                    list(TreeWalker.rec_walk(
                        ast.parse(
                            TreeWalker.read_without_comments(path)
                        )
                    ))
                )
        except:
            walker = TreeWalker([])
        #print(walker.nodes[5])
        return walker.count_everything()

    def matches(self, type):
        if not(type in self.evth1 and type in self.evth2):
            return 0
        s = (len(self.evth1[type]) + len(self.evth2[type])) / 2

        structure_matches = 0
        level_proportion = 0
        level_matches = 0
        ev1, ev2 = self.evth1[type], self.evth2[type]
        for key in ev1:
            if key in ev2:
                structure_matches += 1
                if sorted(ev1[key]) == sorted(ev2[key]):
                    level_proportion += 1
                if ev1[key] == ev2[key]:
                    level_matches += 1
        return {
            'stucture_matches': structure_matches / s,
            'level_proportion': level_proportion / s,
            'level_matches': level_matches / s
        }

    @classmethod
    def read_pairs(cls, input):
        with open(input, 'r') as input:
            pairs = input.readlines()
            for i in range(len(pairs)):
                pairs[i] = pairs[i].strip().split()
        return pairs

    @classmethod
    def load_model(cls, model):
        with open(model if model else Comparer.MODEL, 'rb') as f:
            return pickle.load(f)

    def estimate(self):
        def to_vector():
            vector = []
            cmp = self
            parameters = cmp.matches('body')
            for type in Comparer.TYPES_OF_INTEREST:
                parameters[f'susp_{type}'] = cmp.suspicious_by_type(type)
                parameters[f'overlap_{type}'] = cmp.overlapping_by_type(type)
            for key in sorted(parameters):
                vector.append(parameters[key])
            return vector

        return self.model.predict_proba([to_vector()])[0][1]

    @staticmethod
    def levenstein_distance(s1: str, s2: str) -> int:
        if not s1 and not s2:
            return 0
        elif not s2:
            return len(s1)
        elif not s1:
            return len(s2)

        h, v = len(s1), len(s2)
        upper_row = list(range(h + 1))
        for j in range(v):
            current = [j + 1]
            for i in range(h):
                val = min(
                    upper_row[i] + (s1[i] != s2[j]),
                    upper_row[i + 1] + 1,
                    current[-1] + 1
                )
                current.append(val)
            upper_row = current
        return current[-1]


class WalkResult:
    DROP_IN_STRING = ('col_offset', 'end_lineno', "end_col_offset", 'lineno')

    def __init__(self, type, val, level=0, parent=None, length=0, col_length=0):
        self.type = type
        self.val = val
        self.level = level
        self.parent = parent
        self.length = length
        self.col_length = col_length

    @staticmethod
    def string(obj):
        if isinstance(obj, (str, int, float)):
            return str(obj)
        if isinstance(obj, (list, tuple)):
            return list(map(WalkResult.string, obj))
        if isinstance(obj, ast.Name):
            return obj.id
        if isinstance(obj, ast.Import):
            return str(list(map(WalkResult.string, obj.names)))
        if isinstance(obj, ast.ClassDef):
            return obj.name
        if isinstance(obj, ast.alias):
            return obj.name
        return str(type(obj))[12:-2]


    def __str__(self):
        d = self.__dict__
        for attr in d:
            d[attr] = WalkResult.string(d[attr])
        return str(d)


class TreeWalker:
    def __init__(self, nodes):
        self.nodes = nodes

    @staticmethod
    def read_without_comments(path):
        with open(path, 'rb') as f:
            return re.sub(r'""".*?"""|#.*?\\n', 'pass',
                          str(f.read()).encode('raw_unicode_escape').decode('unicode_escape'))[2:-1]

    @staticmethod
    def rec_walk(ast_node, depth=0):
        for field in ast.iter_fields(ast_node):

            yield WalkResult(*field,
                level=depth,
                parent=ast_node,
                length=(ast_node.end_lineno - ast_node.lineno) if 'end_lineno' in ast_node.__dict__ else -1,
                col_length=(ast_node.end_col_offset - ast_node.col_offset) if 'end_col_offset' in ast_node.__dict__ else -1
            )
        for child in ast.iter_child_nodes(ast_node):
            yield from TreeWalker.rec_walk(child, depth + 1)

    def count_everything(self):
        results = self.nodes
        common_levels = []
        d = {}
        def add_res(result):
            d[result.type] = d.get(result.type, {})
            rv = WalkResult.string(result.val)
            d[result.type][rv] = d[result.type].get(rv, []) + [int(result.level)]
        for result in results:
            common_levels.append(result.level)
            if isinstance(result.val, list):
                for each in result.val:
                    res = WalkResult(result.type, each,
                                     level=result.level,
                                     length=result.length,
                                     col_length=result.col_length
                                     )
                    res.val = each
                    add_res(res)
            else:
                add_res(result)
        return d



if __name__ == '__main__':
    print('Hi')
    parser = create_parser()
    NAMESPACE = parser.parse_args()

    main(NAMESPACE)

    # comparer = Comparer(r'C:\Users\user\PycharmProjects\antiplagiat\plagiat\files\fourier.py',
    #                     r'C:\Users\user\PycharmProjects\antiplagiat\plagiat\plagiat2\fourier.py')
    #
    # print(comparer.estimate())
    #
    # print(*[(i, comparer.evth2[i]) for i in comparer.evth2], sep='\n' )

    # types = set()
    # nodes = []
    # for i in TreeWalker.rec_walk(tree):
    #     print(i)
    #     types.add(i.type)
    #     nodes.append(i)
    # print(types)
    # print(TreeWalker(nodes).count_everything())

    #print(*[i for i in TreeWalker.rec_walk(tree)], sep='\n')
    # for i in os.listdir(r'C:\Users\user\PycharmProjects\antiplagiat\plagiat\files'):
    #     print(f'\n\t{i}\t\n')
    #     paths = {
    #         'files': rf'C:\Users\user\PycharmProjects\antiplagiat\plagiat\files\{i}',
    #         'plag1': rf'C:\Users\user\PycharmProjects\antiplagiat\plagiat\plagiat1\{i}',
    #         'plag2': rf'C:\Users\user\PycharmProjects\antiplagiat\plagiat\plagiat2\{i}'
    #     }
    #
    #
    # # with open('dump.txt', 'w') as d:
    # #     for path in paths:
    # #         print(path, ast.dump(ast.parse(del_comments(paths[path]))), file=d)
    # # with open('initial.txt', 'w') as out:
    # #     print(del_comments(paths['files']), file=out)
    # # with open('wo comments.txt', 'w') as out:
    # #     print(ast.unparse(ast.parse(del_comments(paths['plag2']))), file=out)
    #     txt = del_comments(paths['plag2'])
    #     with open('wo comments.txt', 'w') as out:
    #         print(txt, file=out)
    #     tree = ast.parse(txt)
    #
    #     d = get_vals_by_type(tree)
    #     print(*[f'{i}: {d[i]}' for i in d], sep='\n')
    #     print('_' * 200)


