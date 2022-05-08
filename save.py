from actionutil import combine_actions, action2, RxResp, State
import calculations as calc

def label(state: State) -> RxResp:
    row = state.articles[state.next_article]
    sql = calc.verify_article_sql(row)
    state = state._replace(next_article = state.next_article+1)
    return action2('command_db', sql=sql, status=state.new_label, id=row['RecordId']), state

def classification(state: State) -> RxResp:
    row = state.articles[state.next_article]
    sql = calc.classify_sql()
    auto_class = calc.classify(row)
    total = len(state.articles)
    msg = f"Record: {row['RecordId']} (#{state.next_article} of {total}) Date: {row['PubDate']}, classification: {auto_class}"
    # if auto_class == 'M':
    #     disp, _ = calc.display_article(total, state.next_article, row, ())
    #     msg += f"\n" + disp
    state = state._replace(next_article = state.next_article+1)
    return combine_actions(
        action2('no_op') if auto_class == 'N' else action2('print_message', message = msg),
        action2('command_db', sql=sql, auto_class = auto_class, id = row['RecordId'])
    ), state