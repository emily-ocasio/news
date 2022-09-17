"""
Reactions to prompt GPT3 language model
"""
from state import RxResp, State
from actionutil import action2, next_event
import calculations as calc

pre_article_prompts = {
    'reporter': "A news reporter is having a conversation with his editor. "
        "The reporter presented his draft of an article to the editor. "
        "Here is the news article draft "
        "(there could be multiple unrelated stories mixed together):",
    'article': "The following is a newspaper article that may include "
        "unrelated stories mixed together:"
}

post_article_prompts = {
    '3L_slightly': """The reporter asked the editor: "I want to make sure """
        "that I present the victim $VICTIM in a humanized way. "
        "Do the details in the article, other than the circumstances of "
        "the homicide, show the victim as a person, rather than just a "
        "name, age, and gender? A description of how the victim "
        "was murdered is not evidence of humanizing. "
        "Please respond using the following scoring criteria "
        "for articles:"
        "\nLevel 1: 'Slightly humanizing' "
        "(used for articles that only provide circumstances "
        "of death in addition to age, gender, and location of the victim)"
        "\nLevel 2: 'Somewhat humanizing' (used for articles "
        "that add one or two pieces of information about the victim "
        "other than age, gender, location and circumstances of death)"
        "\nLevel 3: 'Very humanizing' (used for articles that have "
        "extensive details on the victim's life and really shows "
        "them as a person not just a victim)\"\n"
        "\nThe editor was known to be strict, experienced, "
        "and confident, and only considered the portion of the article "
        "dealing with that particular victim. "
        "The editor answered the question directly, and gave a "
        "thoughtful response, justifying the level by examples "
        "specifically about $VICTIM: \"I think this article is Level",
    '3L_not': "The reporter asked the editor: \"I want to make sure "
        "that I present the victim $VICTIM in a humanized way. "
        "Do the details in the article, other than the circumstances of "
        "the homicide, show the victim as a person, rather than just a "
        "name, age, and gender? A description of how the victim "
        "was murdered is not evidence of humanizing. "
        "Please respond using the following scoring criteria "
        "for articles:"
        "\nLevel 1: 'Not humanizing' "
        "(used for articles that only provide circumstances "
        "of death in addition to age, gender, and location of the victim)"
        "\nLevel 2: 'Slightly humanizing' (used for articles "
        "that add one or two pieces of information about the victim "
        "other than age, gender, location and circumstances of death)"
        "\nLevel 3: 'Humanizing enough' (used for articles that have "
        "more details on the victim's life and show "
        "them as a person not just a victim)\"\n"
        "\nThe editor was known to be strict, experienced, "
        "and confident, and only considered the portion of the article "
        "dealing with that particular victim. "
        "The editor answered the question directly, and gave a "
        "thoughtful response, thoroughly justifying the level by examples "
        "specifically about $VICTIM: \"I think this article is Level",
    '3L_not2': "The reporter asked the editor: \"I want to make sure "
        "that I present the victim $VICTIM in a humanized way. "
        "Do the details in the article, other than the circumstances of "
        "the homicide, show the victim as a person, rather than just a "
        "name, age, and gender? A description of how the victim "
        "was murdered is not evidence of humanizing. "
        "Please respond using the following scoring criteria "
        "for articles:"
        "\nLevel 1: 'Not humanizing' "
        "(used for articles that only provide circumstances "
        "of death in addition to age, gender, and location of the victim)"
        "\nLevel 2: 'Slightly humanizing' (used for articles "
        "that add a little information about the victim "
        "other than age, gender, location and circumstances of death)"
        "\nLevel 3: 'Somewhat humanizing' (used for articles that have "
        "more than one detail on the victim's life and show "
        "them as a person not just a victim)\"\n"
        "\nThe editor was known to be strict, experienced, "
        "and confident, and only considered the portion of the article "
        "dealing with that particular victim. "
        "The editor answered the question directly, and gave a "
        "thoughtful response, thoroughly justifying the level by examples "
        "specifically about $VICTIM: \"I think this article is Level",
    '3L_not3': "The reporter asked the editor: \"I want to make sure "
        "that I present the victim $VICTIM in a humanized way. "
        "Do the details in the article, other than the circumstances of "
        "the homicide, show the victim as a person, rather than just a "
        "name, age, and gender? A description of how the victim "
        "was murdered is not evidence of humanizing. "
        "Please respond using the following scoring criteria "
        "for articles:"
        "\nLevel 1: 'Not humanizing' "
        "(used for articles that only provide circumstances "
        "of death in addition to age, gender, and location of the victim)"
        "\nLevel 2: 'Slightly humanizing' (used for articles "
        "that add a little information about the victim "
        "other than age, gender, location and circumstances of death)"
        "but are still not acceptably humanizing"
        "\nLevel 3: 'Somewhat humanizing' (used for articles that have "
        "enough information on the victim's life and show "
        "them as a person not just a victim)\"\n"
        "\nThe editor was known to be experienced "
        "and confident, and only considered the portion of the article "
        "dealing with that particular victim. "
        "The editor answered the question directly, and gave a "
        "thoughtful response, thoroughly justifying the level by examples "
        "specifically about $VICTIM: \"I think this article is Level",
    'multiple': "The article may have information about multiple things. "
        "Extract all the parts of the article that refer only to "
        "the victim $VICTIM:\n",
    'notonly': "The article may have information about multiple topics. "
        "Extract every part of the article that refer specifically to "
        "the victim $VICTIM:\n",
    'every': "Create an extract that includes all the information in the "
        "article that refers to the victim $VICTIM:\n",
    'alsopast': "Create an extract that includes all the information in the "
        "article that refers to the victim $VICTIM, especially any facts "
        "about his life before the incident:\n",
    'alsopast2': "Create an extract that includes all the information in the "
        "article that refers to the victim $VICTIM, especially any facts "
        "about $VICTIM's life before the incident:\n"
}


@next_event('gpt3_responded')
def prompt_gpt(state: State) -> RxResp:
    """
    This is the prompt for the GPT model.
    """
    pre_article = pre_article_prompts[state.pre_article_prompt]
    article = state.articles[state.next_article]['FullText']
    post_article = post_article_prompts[state.post_article_prompt]
    victim = state.homicides_assigned[state.selected_homicide]['Victim']
    prompt, msg = calc.full_gpt3_prompt(pre_article=pre_article,
                post_article=post_article,
                article = article,
                victim = victim)
    return action2('prompt_gpt3', prompt=prompt, msg=msg), state
