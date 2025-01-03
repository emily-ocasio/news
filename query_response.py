"""
Reactions to database queries
"""
from state import RxResp, State
import controller
import calculations as calc


def respond(state: State) -> RxResp:
    """
    Dispatches another function in this module with the name equal to the
        value of state.choice_type
    """
    return globals()[state.query_type](state)


def many_articles(state: State) -> RxResp:
    """
    Response to database query for multiple articles
    """
    state = state._replace(articles=state.outputs)
    return controller.main(state)


def article_types(state: State) -> RxResp:
    """
    Response to query to pull article types for one record
    Occurs when article is to be displayed
    After getting the article types, proceed with displaying full article
    """
    state = state._replace(current_article_types=state.outputs)
    return controller.main(state)


def verified(state: State) -> RxResp:
    """
    Response to query for verified articles
    """
    articles = state.outputs
    matches, nomatches = calc.partition(articles)
    state = state._replace(articles=articles, next_article=0,
                           matches=matches, nomatches=nomatches)
    return controller.main(state)


def all_articles(state: State) -> RxResp:
    """
    Response to query for all articles in a dataset
    """
    articles = state.outputs
    TP, TN, FP, FN = calc.confusion_matrix(articles)
    state = state._replace(
        articles=articles, next_article=0, TP=TP, TN=TN, FP=FP, FN=FN)
    return controller.main(state)


def unclassified_articles(state: State) -> RxResp:
    """
    Response to query for articles to autoclassify
    """
    articles = state.outputs
    state = state._replace(articles=articles, next_article=0)
    return controller.main(state)


def single_article(state: State) -> RxResp:
    """
    Response to query for single article to adjust
    """
    articles = state.outputs
    state = state._replace(articles=articles, next_article=0)
    return controller.main(state)


def unassigned_articles(state: State) -> RxResp:
    """
    Response to query for articles to assign as homicides
    Unassigned articles have been retrieved based on number of days
    """
    articles = state.outputs
    state = state._replace(articles=articles, next_article=0)
    return controller.first_article(state)


def articles_retrieved(state: State) -> RxResp:
    """
    Generic response to multiple articles retrieved
    """
    articles = state.outputs
    state = state._replace(articles=articles, next_article=0,
                            articles_retrieved = True)
    return controller.main(state)


def homicides_retrieved(state: State) -> RxResp:
    """
    Response to multiple homicides retrived
    """
    homicides = state.outputs
    state = state._replace(homicides=homicides,
                            homicides_retrieved=True,
                            current_homicide=-1)
    return controller.main(state)


def auto_assigned_articles(state: State) -> RxResp:
    """
    Response to query for autoclassified articles to be reclassified
    """
    articles = state.outputs
    state = state._replace(articles=articles, next_article=0)
    return controller.first_article(state)


def homicides_by_month(state: State) -> RxResp:
    """
    Response to query for homicides in a particular month
    Also used to query by name or county
    """
    homicides = state.outputs
    state = state._replace(homicides=homicides)
    return controller.main(state)


def assigned_homicides_by_article(state: State) -> RxResp:
    """
    Reponse to query for homicides already assigned to an article
    """
    state = state._replace(homicides_assigned = state.outputs)
    return controller.main(state)


def refreshed_article(state: State) -> RxResp:
    """
    Reset current article row in state after being refreshed from database
    Occurs after changing an article during assignment without proceeding
        to next article (example adding new notes)
    """
    row = state.outputs[0]
    state = state._replace(articles = calc.tuple_replace(
                                            state.articles,
                                            state.next_article,
                                            row),
                            refresh_article = False)
    return controller.main(state)


def refreshed_homicide(state: State) -> RxResp:
    """
    Reset current homicide row in state after being refreshed from database
    Occurs after automatically getting humanizing information for an article
    """
    row = state.outputs[0]
    state = state._replace(homicides = calc.tuple_replace(
                                            state.homicides,
                                            state.current_homicide,
                                            row))
    return controller.main(state)


def articles_to_filter(state: State) -> RxResp:
    """
    Response to query for articles to filter
    """
    state = state._replace(articles=state.outputs, next_article=0)
    return controller.main(state)


def articles_by_victim(state: State) -> RxResp:
    """
    Response to query for articles by victim id
    """
    state = state._replace(articles=state.outputs, next_article=0)
    return controller.main(state)
