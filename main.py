import re
from enum import Enum, auto
from abc import ABC


# List of different token types
class TokenType(Enum):
    # SC Tokens
    PLUS = auto()
    MINUS = auto()
    SLASH = auto()
    STAR = auto()
    LPAREN = auto()
    RPAREN = auto()
    EQUALS = auto()
    MOD = auto()
    GREATER_THAN = auto()
    LESS_THAN = auto()
    BANG = auto()

    # DC Tokens
    BANG_EQUALS = auto()
    EQUALS_EQUALS = auto()
    GREATER_EQUALS = auto()
    LESS_EQUALS = auto()

    # Literals
    IDENTIFIER = auto()
    NUMBER = auto()
    STRING = auto()

    # Keywords
    FALSE = auto()
    TRUE = auto()
    NIL = auto()

    # END
    EOF = auto()


class Token:
    def __init__(self, type: TokenType = None, lexeme: str = "None", literal=None):
        self.type = type
        self.lexeme = lexeme
        self.literal = literal

    def __repr__(self):
        return f"\nType: {self.type.name} \nLexeme: {self.lexeme} \nLiteral:{self.literal}"


# Tokenizes the expression, and creates a list of Token objects
class Scanner:
    def __init__(self, tokenList: list = None):
        self._tokenList = tokenList or []

    def tokenize(self, expression: str) -> list[str]:
        if expression == "":
            return []

        regex = re.compile("\s*(=>|[-+*\/\%=\(\)]|[A-Za-z_][A-Za-z0-9_]*|[0-9]*\.?[0-9]+)\s*")
        tokens = regex.findall(expression)
        # print([s for s in tokens if not s.isspace()])
        return [s for s in tokens if not s.isspace()]

    def addToken(self, type: TokenType, value: str, literal) -> None:
        self._tokenList.append(Token(type=type, lexeme=value, literal=literal))

    def lexer(self, expression: str):
        tokenized = self.tokenize(expression)
        for index, value in enumerate(tokenized):
            if (value.isalpha()):
                self.addToken(type=TokenType.IDENTIFIER, value=value, literal=str)
            elif (value.isdigit()):
                self.addToken(type=TokenType.NUMBER, value=float(value), literal=int)
            elif (value == "+"):
                self.addToken(type=TokenType.PLUS, value=value, literal=None)
            elif (value == "-"):
                self.addToken(type=TokenType.MINUS, value=value, literal=None)
            elif (value == "*"):
                self.addToken(type=TokenType.STAR, value=value, literal=None)
            elif (value == "/"):
                self.addToken(type=TokenType.SLASH, value=value, literal=None)
            elif (value == "%"):
                self.addToken(type=TokenType.MOD, value=value, literal=None)
            elif (value == '='):
                self.addToken(TokenType.EQUALS, value, None)
            elif (value == '('):
                self.addToken(TokenType.LPAREN, value, None)
            elif (value == ')'):
                self.addToken(TokenType.RPAREN, value, None)
        self.addToken(TokenType.EOF, "", None)

        return self._tokenList

    def __repr__(self):
        return f"Token List from Scanner: {self._tokenList}"


# Parent class for other Expression typesf
class Expr:
    pass


# Parent class for other Statement types
class Stmt:
    pass


class Expression(Stmt):
    def __init__(self, expression):
        self.expression = expression

    def accept(self, visitor):
        return visitor.visitExpressionStatement(self)


# Allows for each Node within the tree to be "visited" (The Visitor Pattern)
class ExprVisitor:
    def visitBinary(self, expr):
        pass

    def visitGrouping(self, expr):
        pass

    def visitUnary(self, expr):
        pass

    def visitLiteral(self, expr):
        pass


# Object for Binary expressions (i.e. 1+1)
class Binary(Expr):
    def __init__(self, left: Expr, operator: Token, right: Expr):
        self.left = left
        self.operator = operator
        self.right = right

    def accept(self, visitor):
        return visitor.visitBinary(self)


# Object for Grouping expressions (i.e. "(1+1)")
class Grouping(Expr):
    def __init__(self, expression):
        self.expression = expression

    def accept(self, visitor):
        return visitor.visitGrouping(self)


# Object for Unary expressions (i.e. -(1+1))
class Unary(Expr):
    def __init__(self, operator: Token, right: Expr):
        self.operator = operator
        self.right = right

    def accept(self, visitor):
        return visitor.visitUnary(self)


# Object for Literals (i.e. [0-9])
class Literal(Expr):
    def __init__(self, value: str | int | float):
        self.value = value

    def accept(self, visitor):
        return visitor.visitLiteral(self)


class Var(Stmt):
    def __init__(self, name: Token, initializer):
        self.name = name
        self.initializer = initializer

    def accept(self, visitor):
        return visitor.visitVarStmt(self)


class Variable(Expr):
    def __init__(self, name: Token):
        self.name = name

    def accept(self, visitor):
        return visitor.visitVariableExpr(self)


class ParseError(Exception):
    def __init__(self, message: str = ""):
        Exception(message)


# Parses the list of Tokens -> AST in order
class Parser:
    def __init__(self, tokens: list = [], current: int = 0):
        self.tokens = tokens
        self.current = current

    # Returns the current token
    def peek(self):
        # print(f"Current: {self.current}, {self.tokens[self.current]}")
        return self.tokens[self.current]

    # Checks if current token is EOF
    def isAtEnd(self):
        return self.peek().type == TokenType.EOF

    # Returns the previous token
    def previous(self):
        return self.tokens[self.current - 1]

    # "Consumes" the current token and returns it
    def advance(self):
        # print(f"CALLS ADVANCE")
        if (self.isAtEnd() == False):
            self.current += 1
        return self.previous()

    # Checks if the given token type matches the current token
    def check(self, type: TokenType):
        if (self.isAtEnd()):
            return False
        return self.peek().type == type

    # Checks for a right parenthesis (to match the left) from Token list
    def consume(self, type: TokenType, message: str = ""):
        if (self.check(type)):
            return self.advance()
        else:
            raise ParseError(message)

    # Match: Checks if there's a type match within the list of Tokens
    def match(self, types: list[TokenType]):
        for type in types:
            if (self.check(type)):
                self.advance()
                return True
        return False

    # expression     → equality ;
    def expression(self):
        if (self.isAtEnd()):
            return ''
        # print("CALLS EXPRESSION")
        return self.equality();

    # equality       → comparison ( ( "!=" | "==" ) comparison )*
    def equality(self):
        # print("CALLS EQUALITY")
        expr = self.comparison()
        while (self.match([TokenType.BANG_EQUALS, TokenType.EQUALS_EQUALS])):
            operator = self.previous()
            right = self.comparison()
            expr = Binary(expr, operator, right)
        return expr

    # comparison     → term ( ( ">" | ">=" | "<" | "<=" ) term )* ;
    def comparison(self):
        # print("CALLS COMPARISON")
        expr = self.term()

        while (
        self.match([TokenType.GREATER_THAN, TokenType.LESS_THAN, TokenType.GREATER_EQUALS, TokenType.LESS_EQUALS])):
            operator = self.previous()
            right = self.term()
            expr = Binary(expr, operator, right)
        return expr

    # term           → factor ( ( "-" | "+" ) factor )* ;
    def term(self):
        # print("CALLS TERM")
        expr = self.factor()

        while (self.match([TokenType.MINUS, TokenType.PLUS])):
            # print("FOUND PLUS / MINUS")
            operator = self.previous()
            right = self.factor()
            expr = Binary(expr, operator, right)
        return expr

    # factor         → unary ( ( "/" | "*" | "%" ) unary )* ;
    def factor(self):
        # print("CALLS FACTOR")
        expr = self.unary()
        while (self.match([TokenType.SLASH, TokenType.STAR, TokenType.MOD])):
            # print("FOUND STAR / SLASH / MOD")
            operator = self.previous()
            right = self.unary()
            expr = Binary(expr, operator, right)
        return expr

    # unary          → ( "!" | "-" ) unary | primary ;
    def unary(self):
        # print("CALLS UNARY")
        if (self.match([TokenType.BANG, TokenType.MINUS])):
            # print("FOUND BANG / MINUS")
            operator = self.previous()
            right = self.unary()
            return Unary(operator, right)
        return self.primary()

    # primary  → NUMBER | STRING | "true" | "false" | "nil" | "(" expression ")" ;

    def primary(self):
        # print(f"CALLS PRIMARY")
        if self.match([TokenType.IDENTIFIER]):
            # print(f"FOUND VAR: {Variable(self.previous().lexeme)}")
            return Variable(self.previous().lexeme)
        if (self.match([TokenType.FALSE])):
            return Literal(False)
        if (self.match([TokenType.TRUE])):
            return Literal(True)
        if (self.match([TokenType.NIL])):
            return Literal(None)

        if (self.match([TokenType.NUMBER, TokenType.STRING])):
            # print("FOUND NUMBER / STRING")
            # ASK Jaran
            # print(f"Literal: {self.previous().lexeme}")
            return Literal(self.previous().lexeme)

        if (self.match([TokenType.LPAREN])):
            # print("FOUND LPAREN")
            expr = self.expression()
            self.consume(TokenType.RPAREN, "Expect ) after expression.")
            return Grouping(expr)

        if (self.isAtEnd()):
            raise Exception("Incorrect")

    # def parse(self):
    #     try:
    #         return self.expression()
    #     except:
    #         raise Exception("Cannot Parse")

    # Expression Parsing
    def parse(self):
        if (not self.isAtEnd()):
            statements = self.declaration()
        return statements

    # Checks if declaration is a Variable Declaration or a Statement
    def declaration(self):
        if (self.peek().type == TokenType.IDENTIFIER):
            return self.varDeclaration()
        return self.statement()

    # Creates Variable with initial value
    def varDeclaration(self):
        expr = self.statement()
        # print("VAR DECLARATION")

        initializer = None
        if (self.match([TokenType.EQUALS])):
            initializer = self.expression()
            return Var(expr.expression.name, initializer)
        return expr

    # Selects which statement is entered (IF, PRINT, RETURN)
    def statement(self):
        return self.expressionStatement()

    def expressionStatement(self):
        expr = self.expression()
        return Expression(expr)


# Stores variables from the "environment" in a map
class Environment:
    def __init__(self, dict: dict = dict()):
        self.values = dict

    # Sets variable values
    def define(self, name, value):
        self.values[name] = value

    def retrieve(self, name):
        if (self.values.get(name)):
            return self.values.get(name)
        raise RuntimeError(f"Reference to undeclared variable: {name}")


# Prints out the AST
class PrintAST(ExprVisitor):
    def astPrint(self, expr):
        return expr.accept(self)

    def visitLiteral(self, LiteralExpr: Literal):
        return str(LiteralExpr.value)

    def visitBinary(self, BinaryExpr: Binary):
        return f"({BinaryExpr.operator.lexeme} {BinaryExpr.left.accept(self)} {BinaryExpr.right.accept(self)})"

    def visitUnary(self, UnaryExpr: Unary):
        return f"({UnaryExpr.operator.lexeme} {UnaryExpr.right.accept(self)})"

    def visitGrouping(self, GroupExpr: Grouping):
        return f"(group {GroupExpr.expression.accept(self)})"


# The "visitor" object; Handles the functions for operations
class Interpreter(ExprVisitor):
    def __init__(self):
        self.env = Environment()

    def stringify(self, value):
        if (value == None):
            return "None"
        if (isinstance(value, int) or isinstance(value, float)):
            return str(value)
        else:
            return value

    # Public function for evaluating AST
    def interpret(self, statements):
        try:
            print(statements.accept(self))
            finished = self.execute(statements)
            # print(f"Interpreter: {self.stringify(finished)}")
        except:
            raise RuntimeError()

    # Executes each statement from list (statement?)
    def execute(self, stmt):
        return stmt.accept(self)

    def evaluate(self, expr):
        return expr.accept(self)

    # Checks if value is "truthy" or "falsey"
    def isTruthy(self, value):
        if (value == None):
            return False
        if (isinstance(value, boolean)):
            return value
        return True

        # return False if value == None else value if isinstance(value, boolean) else True

    # Checks if two values are equal
    def isEqual(self, left, right):
        if (left == None and right == None):
            return True
        elif (left == None):
            return False
        else:
            return left == right

    def isNumber(self, operator, left, right):
        try:
            if (isinstance(left, float) or isinstance(left, int)):
                pass
            elif (isinstance(right, float) or isinstance(right, int)):
                pass
            else:
                raise RuntimeError(operator, "Operand(s) must be numbers")
        except:
            raise RuntimeError(operator, "Operand(s) must be numbers")

    # Unary-specific check for int | float
    def isNumberUnary(self, operator, operand):
        try:
            if (isinstance(operand, float) or isinstance(operand, int)):
                pass
            else:
                raise RuntimeError(operator, "Operand must be a number")
        except:
            raise RuntimeError(operator, "Operand must be a number")

    def visitExpressionStatement(self, stmt):
        return self.evaluate(stmt.expression)

    def visitLiteral(self, LiteralExpr: Literal):
        return LiteralExpr.value

    def visitUnary(self, UnaryExpr: Unary):
        right = self.evaluate(UnaryExpr.right)

        match UnaryExpr.operator.type:
            case TokenType.MINUS:
                self.isNumberUnary(UnaryExpr.operator, right)
                return -right
            case TokenType.BANG:
                return not self.isTruthy(right)
            case other:
                return None

    def visitGrouping(self, GroupExpr: Grouping):
        return self.evaluate(GroupExpr.expression)

    def visitVarStmt(self, VarStmt: Var) -> None:
        value = None
        if (VarStmt.initializer != None):
            value = self.evaluate(VarStmt.initializer)
        self.env.define(VarStmt.name, value)
        return value

    def visitVariableExpr(self, VarExpr: Variable):
        return self.env.retrieve(VarExpr.name)

    def visitBinary(self, BinaryExpr: Binary):
        left = self.evaluate(BinaryExpr.left)
        right = self.evaluate(BinaryExpr.right)

        match BinaryExpr.operator.type:
            case TokenType.PLUS:
                try:
                    return left + right
                except:
                    raise RuntimeError(BinaryExpr.operator, "Operands must be two numbers or two strings")
            case TokenType.MINUS:
                self.isNumber(BinaryExpr.operator, left, right)
                return left - right
            case TokenType.STAR:
                self.isNumber(BinaryExpr.operator, left, right)
                return left * right
            case TokenType.SLASH:
                self.isNumber(BinaryExpr.operator, left, right)
                return float(left) / float(right)
            case TokenType.GREATER_THAN:
                self.isNumber(BinaryExpr.operator, left, right)
                return left > right
            case TokenType.LESS_THAN:
                self.isNumber(BinaryExpr.operator, left, right)
                return left < right
            case TokenType.GREATER_EQUALS:
                self.isNumber(BinaryExpr.operator, left, right)
                return left >= right
            case TokenType.LESS_EQUALS:
                self.isNumber(BinaryExpr.operator, left, right)
                return left <= right
            case TokenType.MOD:
                self.isNumber(BinaryExpr.operator, left, right)
                return left % right
            case TokenType.EQUALS_EQUALS:
                return self.isEqual(left, right)
            case TokenType.BANG_EQUALS:
                return not self.isEqual(left, right)


class RuntimeError:
    def __init__(self, token: Token = None, message: str = ""):
        self.token = token
        self.message = message


# Evaluates AST
interpreter = Interpreter()


def run():
    while True:
        try:
            inp = str(input("> "))

            if inp == "quit":
                return

            # AST Printer
            printer = PrintAST()

            # Scans the inputted string and puts each element into Tokens
            scanner = Scanner()
            tokens = scanner.lexer(inp)

            # print(f"Tokens: {tokens}")

            # Parses the list[tokens] and converts to AST
            parser = Parser(tokens)

            # Gets AST
            ans = parser.parse()
            # print(f"Parser Result: {ans}")

            # Prints AST
            # print(f"Parser: {printer.astPrint(ans)}")
            interpreter.interpret(ans)
        except:
            print("Error")
            pass


run()