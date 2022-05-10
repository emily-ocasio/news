from actionutil import combine_actions, action2, RxResp, State
import controller
import calculations as calc

def respond(state: State) -> RxResp:
    """
    Dispatches another function in this module with the name equal to the
        value of state.choice_type
    """
    return globals()[state.query_type](state)

def articles(state: State) -> RxResp:
    """
    Response to database query for multiple articles
    """
    state = state._replace(articles = state.outputs)
    return controller.first_article(state)

def article_types(state: State) -> RxResp:
    """
    Response to query to pull article types for one record
    """
    state = state._replace(current_article_types = state.outputs)
    return controller.show_article(state)

def matches(state: State) -> RxResp:
    articles = state.outputs
    matches, nomatches = calc.partition(articles)
    state = state._replace(articles = articles, next_article = 0, matches = matches, nomatches = nomatches)
    return controller.select_match_group(state)

def all(state: State) -> RxResp:
    articles = state.outputs
    TP, TN, FP, FN = calc.confusion_matrix(articles)
    state = state._replace(articles = articles, next_article = 0, TP = TP, TN = TN, FP = FP, FN = FN)
    return controller.show_statistics(state)

def unclassified_articles(state: State) -> RxResp:
    articles = state.outputs
    state = state._replace(articles = articles, next_article = 0)
    return controller.classify_articles(state)

def single_article(state: State) -> RxResp:
    articles = state.outputs
    state = state._replace(articles = articles, next_article = 0, article_kind = 'review')
    return controller.first_article(state)