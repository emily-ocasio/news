"""
Reactions to prompt GPT3 language model
"""

from state import RxResp, State, HomicideClassResponse
from actionutil import action2, next_event
import calculations as calc


pre_article_prompts = {
    'reporter': "A news reporter is having a conversation with his editor. "
    "The reporter presented his draft of an article to the editor. "
    "Here is the news article draft "
    "(there could be multiple unrelated stories mixed together):\n\n",
    'article': "The following is a newspaper article that may include "
    "unrelated stories mixed together:\n\n",
    'extract': "The following text includes information about "
    "homicide victim $VICTIM\n\n",
    'few-shot': "Articles about homicide victims are humanizing when they "
    "focus on their lives, such as their occupations, relationships and "
    "families. Articles are impersonal when they focus only on what "
    "happened and the criminal's background. An expert editor is asked "
    "to evaluate this article based on these definitions and provide "
    "an opinion on whether the article is humanizing or impersonal.\n"
    "Summary: \"Police made an arrest in an alleged murder yesterday "
    "afternoon. The victim, Raymond Johnson, was found shot once in "
    "the chest with a 38-caliber revolver. Carol Ross was with him at "
    "the time of his murder and will be arraigned tomorrow.\"\n"
    "Response: Impersonal, because it only mentions what happened\n"
    "Summary: \"George Healey was a homicide victim who had been "
    "boarding in the home for 15 years. The 100-pound Healey's "
    "favorite pasttime was fishing in the nearby Connecticut River.\"\n"
    "Response: Humanizing, because it mentions the victim's pastime\n"
    "Summary: ",
    'few-shot2': "Articles about homicide victims are humanizing when they "
    "focus on their lives, such as their occupations, relationships and "
    "families. Articles are impersonal when they focus only on what "
    "happened and the criminal's background. An expert editor is asked "
    "to evaluate this article based on these definitions and provide "
    "an opinion on whether the article is humanizing or impersonal.\n"
    "Summary: \"Raymond Johnson was a man who lived with his companion, "
    "Carol Ross. On the day of his death, he was shot once in the chest "
    "and was found by police on the floor of his apartment.\"\n"
    "Response: Impersonal, because it only mentions what happened\n"
    "Summary: \"George Healey was a homicide victim who had been "
    "boarding in the home for 15 years. The 100-pound Healey's "
    "favorite pasttime was fishing in the nearby Connecticut River.\"\n"
    "Response: Humanizing, because it mentions the victim's pastime\n"
    "Summary: "
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
    "the victim $VICTIM:",
    'every': "Create an extract that includes all the information in the "
    "article that refers to the victim $VICTIM:",
    'alsopast': "\n\nCreate an extract that includes all the information in "
    "the article that refers to the victim $VICTIM, especially any facts "
    "about his life before the incident:",
    'alsopast2': "\n\nCreate an extract that includes all the information in "
    "the article that refers to the victim $VICTIM, especially any facts "
    "about $VICTIM's life before the incident:",
    'alsopast3': "\n\nCreate a text that only includes every piece of "
    "information in the article that refers to the victim $VICTIM, "
    "especially any facts about $VICTIM's life before the incident:",
    'alsopast4': "\n\nCreate a text that only includes every piece of "
    "information in the article that refers to the victim $VICTIM, "
    "especially any facts about $VICTIM's life, occupation, family, and "
    "relationship before the incident:",
    'remove': "\n\nCreate an extract that includes all the information in the "
    "article that refers to the victim $VICTIM, especially any facts "
    "about $VICTIM's life before the incident. Remove from extract "
    "information about age, gender, location, circumstances of death "
    "and information about the suspect and the case:",
    'summary': "\n\nSummarize information after removing age, gender, "
    "location, circumstances of death and information about the suspect "
    "and the case:",
    'rewrite': "\n\nRewrite text after removing age, gender, address, "
    "circumstances of death and information about the suspect and "
    "the investigation:",
    'rewrite2': "\n\nRewrite text after removing age, gender, address, "
    "circumstances of death and information about the suspect:",
    'rewrite3': "\n\nRewrite text after removing age, gender, address, "
    "crime details, and information about the suspect:",
    'rewrite4': "\n\nRewrite text after removing age, gender, address, "
    "and information about the suspect:",
    'few-shot': "\nResponse:",
}

system_prompts = {
    'default': "I will provide a news article from the Boston Globe newspaper. "
    "The article has some words that suggest the topic is homicides or murders,"
    " and place names that suggest it occurred in Massachusetts. "
    "Some articles may include news about more than one incident. "
    "Please categorize the article in one these three cateogires: "
    "(1) an article that refers to a homicide "
    "(not including vehicular homicides) that occurred in Massachussetts, "
    "(2) an article that refers to a homicide that occurred "
    "outside of Massachussetts, "
    "(3) the article does not refer to a homicide. ",
    'homicide_type':
    "I will provide a news article from the Boston Globe newspaper."
    "The article has some words that suggest the topic is a homicide or murder,"
    "Some articles may include news about more than one incident. "
    "Please classify the article into one of the following:\n"
    "'homicide' (this means at least one incident refers to a homicide "
    "not including vehicular homicide or a killing by law enforcement)\n"
    "'vehicular homicide' "
    "(this means the homicide was caused by a traffic accident)\n"
    "'killing by law enforcement' "
    "(this means the killing was done by the police "
    "while responding to someone committing a crime)\n"
    "'fictional homicide' (article mentions homicide "
    "in a book, movie, play, etc., not in real life)\n"
    "'no homicide in article'"
}


@next_event('gpt_responded')
def prompt_gpt(state: State) -> RxResp:
    """
    This is the prompt for the GPT model.
    """
    homicide = (state.homicides[state.current_homicide]
                if state.main_flow == 'humanize'
                else state.homicides_assigned[state.selected_homicide])
    pre_article = pre_article_prompts[state.pre_article_prompt]
    if state.gpt3_source == 'article':
        article = state.articles[state.next_article]['FullText']
    elif state.gpt3_source == 'small':
        article = (state.articles[state.next_article]['SmallExtract']
                   if state.main_flow == 'humanize'
                   else homicide['SmallExtract'])
    else:
        article = (state.articles[state.next_article]['Extract']
                   if state.main_flow == 'humanize'
                   else homicide['Extract'])
    post_article = post_article_prompts[state.post_article_prompt]
    victim = homicide['Victim']
    prompt, msg = calc.full_gpt3_prompt(pre_article=pre_article,
                                        post_article=post_article,
                                        article=article,
                                        victim=victim if victim is not None
                                        else 'unknown')
    return action2('prompt_gpt3', prompt=prompt, msg=msg), state


@next_event('gpt_responded')
def prompt_gpt4(state: State) -> RxResp:
    """ 
    This prompt is for the second filtering of the articles
    """
    system = system_prompts[state.pre_article_prompt]
    article = state.articles[state.next_article]

    user = f'Article Title: """{article["Title"]}"""\n' + \
            f'Article text: """{article["FullText"]}"""'

    return action2('prompt_gpt', system=system,
                   user=user,
                   response_type=HomicideClassResponse), state
