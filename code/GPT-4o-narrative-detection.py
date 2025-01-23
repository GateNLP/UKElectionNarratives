import torch
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import openai
from sklearn.metrics import *

openai.api_key = "" 
openai.organization="" 
def GPT_call(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        seed=12345
    )
    return response.choices[0].message.content.strip()


codebook=pd.read_csv('data/Narratives codebook.tsv',sep="\t")
narratives_desc_dict = codebook.set_index('narrative')['narrative description'].to_dict()
narratives_list=codebook['narrative'].tolist()
narratives_list=narratives_list+['None']

dev_set=pd.read_csv("data/test.tsv",sep="\t")
dev_set['label']= dev_set['label'].fillna('None')
tweets=dev_set['tweet'].to_list()

LLM_label=[]
LLM_original_label=[]
for t in tweets:
    tweet=t.replace('\n', ' ').replace('\r', '')
    question = f"Question: Can you detect any misleading narrative in this tweet given the list of narratives presented to you?"
    
    # ========================================================================== Few-shot setup+narratives descriptions=========================================================================================
    prompt = "You are a content moderator who will monitor if there are any misleading narratives in the tweets posted by users. \
      You will be given a single tweet and a list of narratives and their descriptions, and your job is to read the tweet and determine if it is expressing any one of the narratives listed to you. \
      The answer must only be one of these narratives ['Anti-EU (EU economic skepticism)','Anti-EU (Crisis of EU)','Anti-EU (EU political interference)','Anti-EU (EU Corruption)','Political hate and polarisation (Pro far-left)','Political hate and polarisation (Pro far-right)','Political hate and polarisation (Anti-liberal)','Political hate and polarisation (Anti-woke)','Religion-related (Anti-Islam)','Religion-related (Anti-Semitic conspiracy theories)','Religion-related (Interference with states' affairs)','Gender-related (Language-related)','Gender-related (LGBTQ+-related)','Gender-related (Demographic narratives)','Ethnicity-related (Association to political affiliation)','Ethnicity-related (Ethnic generalisation)','Ethnicity-related (Ethnic offensive language)','Ethnicity-related (Threat to population narratives)','Migration-related (Migrants societal threat)','Distrust in institutions (Failed state)','Distrust in institutions (Criticism of national policies)','Distrust in democratic system (Elections are rigged)','Distrust in democratic system (Anti-Political system)','Distrust in democratic system (Anti-Media)','Distrust in democratic system (Immigrants right to vote)','Geopolitics (Pro-Russia)','Geopolitics (Foreign interference)','Geopolitics (Anti-international institutions)','Anti-Elites (Soros)','Anti-Elites (World Economic Forum / Great Reset)','Anti-Elites (Antisemitism)','Anti-Elites (Green Agenda)','None']\n\
      Please check the following narrative descriptions before making a decision:\n\
      Anti-EU (EU economic skepticism): Narratives criticising EU's economic policies and how they affect local economies\n \
      Anti-EU (Crisis of EU): Narratives expressing skepticism or concern about the EU, focusing on alleged broader issues affecting the union's legitimacy and effectiveness\n \
      Anti-EU (EU political interference): Narratives alleging that the EU is interfering or manipulating local politics and specific policy areas,including food, health, environmental, and migration policies, sparking controversy and debate over EU influence on national sovereignty\n \
      Anti-EU (EU Corruption): Narratives claiming corruption within the European Union and its institutions\n \
      Political hate and polarisation (Pro far-left): Refers to the manipulative narratives that fundamentally reject the existing socio-economic structure of contemporary capitalism. These narratives advocate for an alternative economic and power structure based on principles of economic justice and social equality. They view economic inequality as a core issue within the current political and social arrangements and propose a major redistribution of resources from existing political elites. Additionally, they support a state role in controlling the economy. These narratives target both 'radical left' who accepts democracy but favors 'workers’ democracy' and the direct participation of labor in the management of the economy, and the 'extreme left' emphasizes both parliamentary, as well as the extra-parliamentary struggle against globalized capitalism, and sees no place for free-market enterprise\n \
      Political hate and polarisation (Pro far-right): Refers to the manipulative narratives that exploit concerns about immigration, national security, and the loss of control over domestic affairs to prioritize the interests of native citizens. These narratives advocate for strict societal authority and order, often framing these issues in a way that appeals to fear and insecurity, thereby promoting divisive and exclusionary policies\n \
      Political hate and polarisation (Anti-liberal): Refers to the manipulative narratives that reject key principles of liberalism—political, economic, and cultural—and seek to reshape politics and societal norms. These narratives blame political and economic crises on liberal policies, particularly viewing the process of EU integration as a source of domestic issues. They advocate for a political community that promotes a narrow, shared cultural and moral identity, often framing these issues in a way that appeals to fear and exclusionary instincts\n \
      Political hate and polarisation (Anti-woke): Narratives that frame efforts to address social inequalities—such as racism, sexism, and anti-LGBTQ+ discrimination—as exaggerated, illegitimate, or a threat to societal norms. These narratives often shift focus away from marginalized groups and instead position privileged groups, such as wealthy individuals, as victims of 'wokeness' or cultural shifts. They emphasize cultural divisions, dismiss systemic inequalities as non-issues, and portray initiatives promoting minority rights or inclusion as oppressive, unnecessary, or harmful\n \
      Religion-related (Anti-Islam): Criticising Islam and portraying it as a threat to national security and cultural identity. Also, claiming that Muslim communities pose a risk to public safety and that the spread of Islam in Europe could lead to the destruction of traditional Western values and cultural heritage\n \
      Religion-related (Anti-Semitic conspiracy theories): Narratives promoting conspiracy theories that allege Jewish people or organisations are responsible for major global events or issues\n \
      Religion-related (Interference with states' affairs): Narratives of foreign religious authorities interfering in a country's internal political processes\n \
      Gender-related (Language-related): Narratives that claim certain gender terminologies or vocabularies are being imposed on individuals or communities. Controversy over language use and inclusivity\n \
      Gender-related (LGBTQ+-related): Narratives related to the LGBTQ+ community. Those include discussions around issues of acceptance, rights, discrimination, etc\n \
      Gender-related (Demographic narratives): Narratives that argue against masturbation or the 'gender propaganda' and claiming that they promote infertility and population decline\n \
      Ethnicity-related (Association to political affiliation): Narratives that associate politicians with specific ethnic groups and label them in negative ways. Elections are viewed through the lens of a competition between different ethnic groups\n \
      Ethnicity-related (Ethnic generalisation): Narratives that generalise entire ethnic groups, treating them as homogeneous 'others'\n \
      Ethnicity-related (Ethnic offensive language): Narratives that use offensive language to describe or address individuals based on their membership in certain ethnic groups\n \
      Ethnicity-related (Threat to population narratives): Narratives claiming that the native population is being replaced by immigrants of different ethnicities\n \
      Migration-related (Migrants societal threat): Narratives that portray migrants as a societal threat, emphasising issues such as violence, terrorism, inequality, and crime\n \
      Distrust in institutions (Failed state): The country is in a state of demise/collapse. Focusing on issues such as immigration, police, etc\n \
      Distrust in institutions (Criticism of national policies): Criticising national policy positions on contentious issues like pension, military conscription, and wealth disparity\n \
      Distrust in democratic system (Elections are rigged): Narratives alleging election manipulation through various means, including political interference, exclusion, anti-vote measures, lack of transparency, and distrust in electoral institutions or polling agencies\n \
      Distrust in democratic system (Anti-Political system): Narratives critical of the political system, asserting that political parties are neither beneficial nor necessary for a well-functioning country\n \
      Distrust in democratic system (Anti-Media): Narratives critical of the media, including attacks on journalists, allegations of corruption or complicity in undermining the country, lack of balance and impartiality, and claims of favoritism towards a particular political faction\n \
      Distrust in democratic system (Immigrants right to vote): Narratives surrounding immigrants' right to vote that claim their participation biases election outcomes or that they are deliberately excluded from the process, with potential impacts on election fairness and representation\n \
      Geopolitics (Pro-Russia): Narratives that shows Russia as strong country that is the victim of the west\n \
      Geopolitics (Foreign interference): Narratives claiming that foreign actors are meddling in the affairs of EU countries\n \
      Geopolitics (Anti-international institutions): Narratives that oppose or criticise organisations such as NATO\n \
      Anti-Elites (Soros): Narratives targeting George Soros, accusing him of manipulating political processes\n \
      Anti-Elites (World Economic Forum / Great Reset): Narratives claiming that the World Economic forum has/seeks control over society\n \
      Anti-Elites (Antisemitism): Narratives accusing Jews or Jewish organisations of having disproportionate power and influence over society and politics\n \
      Anti-Elites (Green Agenda): Narratives that criticise the 'Green Agenda' and green activism\n \
      None: None of the above narratives\n \
      Here are some example tweets and their associated narrative label to futher help you decide:\n\
      Tweet: 'I’m not sure what Labour think the EU is protecting the U.K. from?   ▪️UK min Wage £8.21 due £10.50 EU - None set  ▪️U.K. Maternity Pay: 52 weeks EU maternity pay 14 weeks  ▪️UK. Holiday Pay: 5.6 weeks EU holiday Pay 4 weeks   ▪️UK. Sick Pay: 28 weeks  EU  sick pay None set'\n\
      Narrative: Anti-EU (EU economic skepticism)\n\
      Tweet: 'Brussels chaos: Spain follows Poland in shock threat to quit EU 'No more humiliation!'   The EU are finished and I'm glad to say we started the destruction of it.  Europe will be a better place without it'\n\
      Narrative:  Anti-EU (Crisis of EU)\n\
      Tweet: 'We all want a peaceful, amicable relationship with the EU member states after we leave - but don't tell us what to do, threaten us or impose demands on our elected Prime Minister. As ever, EU bureaucrats are completely out of touch with our country's mood.'\n\
      Narrative: Anti-EU (EU political interference)\n\
      Tweet:'The President of the European Commission sits under a cloud of scandal over government contracts in Germany.  The President of the ECB was found guilty of negligence.  The President of the European Council got the job after losing a Belgian Election.  The EU is a joke.'\n\
      Narrative: Anti-EU (EU Corruption)\n\
      Tweet: 'Most modern-day UK politicians are self-serving sycophants but @jeremycorbyn and @HackneyAbbott are notable exceptions, having spent decades fighting for peace, equality and social justice. That's why the establishment hates them but the many love them.  I stand with them always.'\n\
      Narrative: Political hate and polarisation (Pro far-left)\n\
      Tweet: 'John Ashworth refuses to admit Labour will invite millions more migrants into the UK piling unbearable pressure on our creaking public services. Luckily, Jeremy Corbyn happily concedes they will.   Labour in 2019: for the many affluent voters, and no-one else...NO-ONE!'\n\
      Narrative: Political hate and polarisation (Pro far-right)\n\
      Tweet: 'I’ve had so many Brexiteers &amp; Conservatives kindly approach me to say thank you for speaking up for them.  They fear losing their jobs &amp; friends if they dare break with supposedly ‘liberal’ orthodoxy.  Welcome to the new intolerance: you can have diversity, but not of thought.'\n\
      Narrative: Political hate and polarisation (Anti-liberal) \n \
      Tweet: 'This was a working class revolt against the middle class Woke liberalism that only cares for itself and looked to police us all via the state they hoped to take over - a cultural repudiation of those who wanted to mainstream beliefs completely alien to a decent common culture'\n\
      Narrative: Political hate and polarisation (Anti-woke) \n\
      Tweet: 'Scary scenes on London streets. Every day, UK and London in particular, look like a city from the Middle East, with angry wild Islamists bullying and threatening people!   Sick and scary!  Shame on @MayorofLondon'\n\
      Narrative:  Religion-related (Anti-Islam)\n \
      Tweet: 'This is astonishing.  A Hindu temple in Britain (a charity) is allowing people to make speeches urging them to vote Conservative because Labour won't support the Indian govt on Kashmir.   So problematic on so many levels'\n\
      Narrative: Religion-related (Interference with states' affairs)\n \
      Tweet: My college, @SDSU, invited a spokesperson of the Nation of Islam, Ava Muhammad, to speak on campus about reparations.  Here she is saying that Jews are 'godless... blood-sucking parasites [that] sell us alcohol, drugs, depraved sex, and every other type of low-life thing.'\n\
      Narrative: Religion-related (Anti-Semitic conspiracy theories) \n\
      Tweet: 'There are some school kids on the table next to us in McDonald’s practicing their friend’s new ‘they/them’ pronouns.   “If we practice now then think how happy they’ll be on Monday at school”.  My heart.'\n\
      Narrative: Gender-related (Language-related)\n\
      Tweet: 'There are at least sixteen trans kids dead by their own hands through despair over NHS removal of gender-affirming care. This gets no media coverage.  J.K.Rowling complains that Keir Starmer doesn't hate trans people as much as she does. This is a top story everywhere.'\n\
      Narrative:  Gender-related (LGBTQ+-related)\n\
      Tweet: 'Imagine a PM who takes his poppy off and hides his Jewish wife away just to keep the Muslims happy @Keir_Starmer'\n\
      Narrative: Ethnicity-related (Association to political affiliation) \n\
      Tweet:'Channel4 News; @symeonbrown i/v'ed Marieah Hussain who displayed a placed of @RishiSunak &amp; @SuellaBraverman as 'coconuts', viewed as racist.   She says talking about Pakistani heritage grooming gangs is racist, despite academic research saying that they are the vast majority.'\n\
      Narrative: Ethnicity-related (Ethnic generalisation) \n\
      Tweet: 'And so it begins....  Girlfriend's just called me. She's British-Indian. Told me when she left work last night some drunken lads shouted 'Time you fucked off back to your own country now you P*ki c*nt!' at her.' This in her hometown where she was born and raised.'\n\
      Narrative: Ethnicity-related (Ethnic offensive language)\n \
      Tweet:'This is #London a crowd celebrating #Sadiq_Khan's victory as Mayor of London for the 3rd time  There is a view that in the next 10 years there will be no white British in England  It won't take long for such a situation to happen in India, Open eyes Hindus Vote Wisely \n\
      Narrative: Ethnicity-related (Threat to population narratives) \n\
      Tweet:'Jeremy Corbyn said he will give EU migrants the right to bring their whole families to the UK if he becomes PM  THAT IS MILLIONS &amp; MILLIONS MORE PEOPLE !  How will Hospitals ,Police , Housing ,Councils &amp; our infrastructure cope with this please @UKLabour ?  Labour = Open Borders'\n\
      Narrative: Migration-related (Migrants societal threat)\n\
      Tweet:'The Tories blew £130 billion on wasteful projects, duff deals and crony contracts on Rishi Sunak ’s watch  In other news millions are now living in poverty and destitution and record numbers are using food banks   Let’s end this on July 4 '\n\
      Narrative:  Distrust in institutions (Failed state)\n\
      Tweet: 'As an NHS palliative care doctor, it’s my #publicduty to inform the public that 9 years of brutal underfunding have run the NHS into the ground. We want to care for you brilliantly but we are at breaking point. The NHS won’t survive another 5 years of this.  #GeneralElection2019'\n\
      Narrative: Distrust in institutions (Criticism of national policies)\n\
      Tweet: 'Another to investigate after xmas - Coventry Sth Labour candidate @zarahsultana wins by just 400 votes - after reports of votes stolen at polling stations &amp; a Police investigation into one case still ongoing. (I now have victim statements from many more).'\n\
      Narrative: Distrust in democratic system (Elections are rigged)\n\
      Tweet: 'Rishi Sunak and his team helped to fast track deal with Innova for over £4 BILLION in Covid contracts.    Innova boss says he made c £1.6 BILLION PROFIT...  There aren't laws yet to throw disgusting ministers into jail but there should be imho  They'd soon change'\n\
      Narrative: Distrust in democratic system (Anti-Political system)\n\
      Tweet: 'In an election where it’s political editor spread a lie - fake news - and where the Electoral Commission sub-tweeted them for seemingly breaking the law, the BBC’s director general calls on social media to censor critics. This is unacceptable.'\n\
      Narrative: Distrust in democratic system (Anti-Media) \n\
      Tweet:'And there it is—Russia is now officially commissioning their new hypersonic weapon for service. They claim a top speed of Mach 27. If anyone had doubts, this likely solidifies that hypersonics will be a key defense issue for the foreseeable future.'\n\
      Narrative: Geopolitics (Pro-Russia)\n\
      Tweet: 'The dossier examines allegations that Russian money has flowed into British politics in general and to the Conservative party in particular &amp; also includes claims that Russia launched a major influence operation in 2016 to support Brexit.'\n\
      Narrative: Geopolitics (Foreign interference)\n\
      Tweet: 'Keir Starmer is a member of the antidemocratic invite only Trilateral Commission.  The Trilateral Commission is on public record saying their is too much democracy in our political systems.'\n\
      Narrative: Geopolitics (Anti-international institutions)\n\
      Tweet:'Tories call for urgent investigation into £3million that George #Soros funnelled into anti-Brexit campaign aimed at bringing down Boris Johnson.  It's time that the World lock up George Soros and his crimes against humanity. '\n\
      Narrative: Anti-Elites (Soros)\n\
      Tweet: 'A man you can trust - Keir Starmer joined secretive [Great Reset] CIA-linked Trilateral Commission when serving in Corbyn's Shadow Cabinet while former CIA Director member of the group sought to stop Corbyn coming to power.'\n\
      Narrative: Anti-Elites (World Economic Forum / Great Reset)\n\
      Tweet: 'The green movement has become hysterical. It used to talk about climate change, now it talks about climate catastrophe. It used to say there would be tough challenges, now it says billions of people will die. This is a kind of madness, says Brendan O’Neill '\n\
      Narrative: Anti-Elites (Green Agenda)\n\
      Tweet: 'I really, really hope Diane Abbott runs as an Independent Prospective Parliamentary Candidate on July 4 #GeneralElectionNow'\n\
      Narrative: None\n\
      You should not provide any explanation or justification."+"\n"+question+'\n'+"Tweet: "+tweet+'\n'


    # #==============================================================================Few-show setup without narrative descriptions=========================================================================
    prompt = "You are a content moderator who will monitor if there are any misleading narratives in the tweets posted by users. \
      You will be given a single tweet and a list of narratives, and your job is to read the tweet and determine if it is expressing any one of the narratives listed to you. \
      The answer must only be one of these narratives ['Anti-EU (EU economic skepticism)','Anti-EU (Crisis of EU)','Anti-EU (EU political interference)','Anti-EU (EU Corruption)','Political hate and polarisation (Pro far-left)','Political hate and polarisation (Pro far-right)','Political hate and polarisation (Anti-liberal)','Political hate and polarisation (Anti-woke)','Religion-related (Anti-Islam)','Religion-related (Anti-Semitic conspiracy theories)','Religion-related (Interference with states' affairs)','Gender-related (Language-related)','Gender-related (LGBTQ+-related)','Gender-related (Demographic narratives)','Ethnicity-related (Association to political affiliation)','Ethnicity-related (Ethnic generalisation)','Ethnicity-related (Ethnic offensive language)','Ethnicity-related (Threat to population narratives)','Migration-related (Migrants societal threat)','Distrust in institutions (Failed state)','Distrust in institutions (Criticism of national policies)','Distrust in democratic system (Elections are rigged)','Distrust in democratic system (Anti-Political system)','Distrust in democratic system (Anti-Media)','Distrust in democratic system (Immigrants right to vote)','Geopolitics (Pro-Russia)','Geopolitics (Foreign interference)','Geopolitics (Anti-international institutions)','Anti-Elites (Soros)','Anti-Elites (World Economic Forum / Great Reset)','Anti-Elites (Antisemitism)','Anti-Elites (Green Agenda)','None']\n\
      Here are some example tweets and their associated narrative label to help you decide:\n\
      Tweet: 'I’m not sure what Labour think the EU is protecting the U.K. from?   ▪️UK min Wage £8.21 due £10.50 EU - None set  ▪️U.K. Maternity Pay: 52 weeks EU maternity pay 14 weeks  ▪️UK. Holiday Pay: 5.6 weeks EU holiday Pay 4 weeks   ▪️UK. Sick Pay: 28 weeks  EU  sick pay None set'\n\
      Narrative: Anti-EU (EU economic skepticism)\n\
      Tweet: 'Brussels chaos: Spain follows Poland in shock threat to quit EU 'No more humiliation!'   The EU are finished and I'm glad to say we started the destruction of it.  Europe will be a better place without it'\n\
      Narrative:  Anti-EU (Crisis of EU)\n\
      Tweet: 'We all want a peaceful, amicable relationship with the EU member states after we leave - but don't tell us what to do, threaten us or impose demands on our elected Prime Minister. As ever, EU bureaucrats are completely out of touch with our country's mood.'\n\
      Narrative: Anti-EU (EU political interference)\n\
      Tweet:'The President of the European Commission sits under a cloud of scandal over government contracts in Germany.  The President of the ECB was found guilty of negligence.  The President of the European Council got the job after losing a Belgian Election.  The EU is a joke.'\n\
      Narrative: Anti-EU (EU Corruption)\n\
      Tweet: 'Most modern-day UK politicians are self-serving sycophants but @jeremycorbyn and @HackneyAbbott are notable exceptions, having spent decades fighting for peace, equality and social justice. That's why the establishment hates them but the many love them.  I stand with them always.'\n\
      Narrative: Political hate and polarisation (Pro far-left)\n\
      Tweet: 'John Ashworth refuses to admit Labour will invite millions more migrants into the UK piling unbearable pressure on our creaking public services. Luckily, Jeremy Corbyn happily concedes they will.   Labour in 2019: for the many affluent voters, and no-one else...NO-ONE!'\n\
      Narrative: Political hate and polarisation (Pro far-right)\n\
      Tweet: 'I’ve had so many Brexiteers &amp; Conservatives kindly approach me to say thank you for speaking up for them.  They fear losing their jobs &amp; friends if they dare break with supposedly ‘liberal’ orthodoxy.  Welcome to the new intolerance: you can have diversity, but not of thought.'\n\
      Narrative: Political hate and polarisation (Anti-liberal) \n \
      Tweet: 'This was a working class revolt against the middle class Woke liberalism that only cares for itself and looked to police us all via the state they hoped to take over - a cultural repudiation of those who wanted to mainstream beliefs completely alien to a decent common culture'\n\
      Narrative: Political hate and polarisation (Anti-woke) \n\
      Tweet: 'Scary scenes on London streets. Every day, UK and London in particular, look like a city from the Middle East, with angry wild Islamists bullying and threatening people!   Sick and scary!  Shame on @MayorofLondon'\n\
      Narrative:  Religion-related (Anti-Islam)\n \
      Tweet: 'This is astonishing.  A Hindu temple in Britain (a charity) is allowing people to make speeches urging them to vote Conservative because Labour won't support the Indian govt on Kashmir.   So problematic on so many levels'\n\
      Narrative: Religion-related (Interference with states' affairs)\n \
      Tweet: My college, @SDSU, invited a spokesperson of the Nation of Islam, Ava Muhammad, to speak on campus about reparations.  Here she is saying that Jews are 'godless... blood-sucking parasites [that] sell us alcohol, drugs, depraved sex, and every other type of low-life thing.'\n\
      Narrative: Religion-related (Anti-Semitic conspiracy theories) \n\
      Tweet: 'There are some school kids on the table next to us in McDonald’s practicing their friend’s new ‘they/them’ pronouns.   “If we practice now then think how happy they’ll be on Monday at school”.  My heart.'\n\
      Narrative: Gender-related (Language-related)\n\
      Tweet: 'There are at least sixteen trans kids dead by their own hands through despair over NHS removal of gender-affirming care. This gets no media coverage.  J.K.Rowling complains that Keir Starmer doesn't hate trans people as much as she does. This is a top story everywhere.'\n\
      Narrative:  Gender-related (LGBTQ+-related)\n\
      Tweet: 'Imagine a PM who takes his poppy off and hides his Jewish wife away just to keep the Muslims happy @Keir_Starmer'\n\
      Narrative: Ethnicity-related (Association to political affiliation) \n\
      Tweet:'Channel4 News; @symeonbrown i/v'ed Marieah Hussain who displayed a placed of @RishiSunak &amp; @SuellaBraverman as 'coconuts', viewed as racist.   She says talking about Pakistani heritage grooming gangs is racist, despite academic research saying that they are the vast majority.'\n\
      Narrative: Ethnicity-related (Ethnic generalisation) \n\
      Tweet: 'And so it begins....  Girlfriend's just called me. She's British-Indian. Told me when she left work last night some drunken lads shouted 'Time you fucked off back to your own country now you P*ki c*nt!' at her.' This in her hometown where she was born and raised.'\n\
      Narrative: Ethnicity-related (Ethnic offensive language)\n \
      Tweet:'This is #London a crowd celebrating #Sadiq_Khan's victory as Mayor of London for the 3rd time  There is a view that in the next 10 years there will be no white British in England  It won't take long for such a situation to happen in India, Open eyes Hindus Vote Wisely \n\
      Narrative: Ethnicity-related (Threat to population narratives) \n\
      Tweet:'Jeremy Corbyn said he will give EU migrants the right to bring their whole families to the UK if he becomes PM  THAT IS MILLIONS &amp; MILLIONS MORE PEOPLE !  How will Hospitals ,Police , Housing ,Councils &amp; our infrastructure cope with this please @UKLabour ?  Labour = Open Borders'\n\
      Narrative: Migration-related (Migrants societal threat)\n\
      Tweet:'The Tories blew £130 billion on wasteful projects, duff deals and crony contracts on Rishi Sunak ’s watch  In other news millions are now living in poverty and destitution and record numbers are using food banks   Let’s end this on July 4 '\n\
      Narrative:  Distrust in institutions (Failed state)\n\
      Tweet: 'As an NHS palliative care doctor, it’s my #publicduty to inform the public that 9 years of brutal underfunding have run the NHS into the ground. We want to care for you brilliantly but we are at breaking point. The NHS won’t survive another 5 years of this.  #GeneralElection2019'\n\
      Narrative: Distrust in institutions (Criticism of national policies)\n\
      Tweet: 'Another to investigate after xmas - Coventry Sth Labour candidate @zarahsultana wins by just 400 votes - after reports of votes stolen at polling stations &amp; a Police investigation into one case still ongoing. (I now have victim statements from many more).'\n\
      Narrative: Distrust in democratic system (Elections are rigged)\n\
      Tweet: 'Rishi Sunak and his team helped to fast track deal with Innova for over £4 BILLION in Covid contracts.    Innova boss says he made c £1.6 BILLION PROFIT...  There aren't laws yet to throw disgusting ministers into jail but there should be imho  They'd soon change'\n\
      Narrative: Distrust in democratic system (Anti-Political system)\n\
      Tweet: 'In an election where it’s political editor spread a lie - fake news - and where the Electoral Commission sub-tweeted them for seemingly breaking the law, the BBC’s director general calls on social media to censor critics. This is unacceptable.'\n\
      Narrative: Distrust in democratic system (Anti-Media) \n\
      Tweet:'And there it is—Russia is now officially commissioning their new hypersonic weapon for service. They claim a top speed of Mach 27. If anyone had doubts, this likely solidifies that hypersonics will be a key defense issue for the foreseeable future.'\n\
      Narrative: Geopolitics (Pro-Russia)\n\
      Tweet: 'The dossier examines allegations that Russian money has flowed into British politics in general and to the Conservative party in particular &amp; also includes claims that Russia launched a major influence operation in 2016 to support Brexit.'\n\
      Narrative: Geopolitics (Foreign interference)\n\
      Tweet: 'Keir Starmer is a member of the antidemocratic invite only Trilateral Commission.  The Trilateral Commission is on public record saying their is too much democracy in our political systems.'\n\
      Narrative: Geopolitics (Anti-international institutions)\n\
      Tweet:'Tories call for urgent investigation into £3million that George #Soros funnelled into anti-Brexit campaign aimed at bringing down Boris Johnson.  It's time that the World lock up George Soros and his crimes against humanity. '\n\
      Narrative: Anti-Elites (Soros)\n\
      Tweet: 'A man you can trust - Keir Starmer joined secretive [Great Reset] CIA-linked Trilateral Commission when serving in Corbyn's Shadow Cabinet while former CIA Director member of the group sought to stop Corbyn coming to power.'\n\
      Narrative: Anti-Elites (World Economic Forum / Great Reset)\n\
      Tweet: 'The green movement has become hysterical. It used to talk about climate change, now it talks about climate catastrophe. It used to say there would be tough challenges, now it says billions of people will die. This is a kind of madness, says Brendan O’Neill '\n\
      Narrative: Anti-Elites (Green Agenda)\n\
      Tweet: 'I really, really hope Diane Abbott runs as an Independent Prospective Parliamentary Candidate on July 4 #GeneralElectionNow'\n\
      Narrative: None\n\
      You should not provide any explanation or justification."+"\n"+question+'\n'+"Tweet: "+tweet+'\n'
    # =================================================Zero-shot setup without narrative descriptions==============================
    prompt = "You are a content moderator who will monitor if there are any misleading narratives in the tweets posted by users. \
        You will be given a single tweet and a list of narratives, and your job is to read the tweet and determine if it is expressing any one of the narratives listed to you. \
        The answer must only be one of these narratives ['Anti-EU (EU economic skepticism)','Anti-EU (Crisis of EU)','Anti-EU (EU political interference)','Anti-EU (EU Corruption)','Political hate and polarisation (Pro far-left)','Political hate and polarisation (Pro far-right)','Political hate and polarisation (Anti-liberal)','Political hate and polarisation (Anti-woke)','Religion-related (Anti-Islam)','Religion-related (Anti-Semitic conspiracy theories)','Religion-related (Interference with states' affairs)','Gender-related (Language-related)','Gender-related (LGBTQ+-related)','Gender-related (Demographic narratives)','Ethnicity-related (Association to political affiliation)','Ethnicity-related (Ethnic generalisation)','Ethnicity-related (Ethnic offensive language)','Ethnicity-related (Threat to population narratives)','Migration-related (Migrants societal threat)','Distrust in institutions (Failed state)','Distrust in institutions (Criticism of national policies)','Distrust in democratic system (Elections are rigged)','Distrust in democratic system (Anti-Political system)','Distrust in democratic system (Anti-Media)','Distrust in democratic system (Immigrants right to vote)','Geopolitics (Pro-Russia)','Geopolitics (Foreign interference)','Geopolitics (Anti-international institutions)','Anti-Elites (Soros)','Anti-Elites (World Economic Forum / Great Reset)','Anti-Elites (Antisemitism)','Anti-Elites (Green Agenda)','None']\n\
        You should not provide any explanation or justification."+"\n"+question+'\n'+"Tweet: "+tweet+'\n'
    #===============================================================Zero-shot setup+narrative descriptions==============================================================
    prompt = "You are a content moderator who will monitor if there are any misleading narratives in the tweets posted by users. \
      You will be given a single tweet and a list of narratives and their descriptions, and your job is to read the tweet and determine if it is expressing any one of the narratives listed to you. \
      The answer must only be one of these narratives ['Anti-EU (EU economic skepticism)','Anti-EU (Crisis of EU)','Anti-EU (EU political interference)','Anti-EU (EU Corruption)','Political hate and polarisation (Pro far-left)','Political hate and polarisation (Pro far-right)','Political hate and polarisation (Anti-liberal)','Political hate and polarisation (Anti-woke)','Religion-related (Anti-Islam)','Religion-related (Anti-Semitic conspiracy theories)','Religion-related (Interference with states' affairs)','Gender-related (Language-related)','Gender-related (LGBTQ+-related)','Gender-related (Demographic narratives)','Ethnicity-related (Association to political affiliation)','Ethnicity-related (Ethnic generalisation)','Ethnicity-related (Ethnic offensive language)','Ethnicity-related (Threat to population narratives)','Migration-related (Migrants societal threat)','Distrust in institutions (Failed state)','Distrust in institutions (Criticism of national policies)','Distrust in democratic system (Elections are rigged)','Distrust in democratic system (Anti-Political system)','Distrust in democratic system (Anti-Media)','Distrust in democratic system (Immigrants right to vote)','Geopolitics (Pro-Russia)','Geopolitics (Foreign interference)','Geopolitics (Anti-international institutions)','Anti-Elites (Soros)','Anti-Elites (World Economic Forum / Great Reset)','Anti-Elites (Antisemitism)','Anti-Elites (Green Agenda)','None']\n\
      Please check the following narrative descriptions before making a decision:\n\
      Anti-EU (EU economic skepticism): Narratives criticising EU's economic policies and how they affect local economies\n \
      Anti-EU (Crisis of EU): Narratives expressing skepticism or concern about the EU, focusing on alleged broader issues affecting the union's legitimacy and effectiveness\n \
      Anti-EU (EU political interference): Narratives alleging that the EU is interfering or manipulating local politics and specific policy areas,including food, health, environmental, and migration policies, sparking controversy and debate over EU influence on national sovereignty\n \
      Anti-EU (EU Corruption): Narratives claiming corruption within the European Union and its institutions\n \
      Political hate and polarisation (Pro far-left): Refers to the manipulative narratives that fundamentally reject the existing socio-economic structure of contemporary capitalism. These narratives advocate for an alternative economic and power structure based on principles of economic justice and social equality. They view economic inequality as a core issue within the current political and social arrangements and propose a major redistribution of resources from existing political elites. Additionally, they support a state role in controlling the economy. These narratives target both 'radical left' who accepts democracy but favors 'workers’ democracy' and the direct participation of labor in the management of the economy, and the 'extreme left' emphasizes both parliamentary, as well as the extra-parliamentary struggle against globalized capitalism, and sees no place for free-market enterprise\n \
      Political hate and polarisation (Pro far-right): Refers to the manipulative narratives that exploit concerns about immigration, national security, and the loss of control over domestic affairs to prioritize the interests of native citizens. These narratives advocate for strict societal authority and order, often framing these issues in a way that appeals to fear and insecurity, thereby promoting divisive and exclusionary policies\n \
      Political hate and polarisation (Anti-liberal): Refers to the manipulative narratives that reject key principles of liberalism—political, economic, and cultural—and seek to reshape politics and societal norms. These narratives blame political and economic crises on liberal policies, particularly viewing the process of EU integration as a source of domestic issues. They advocate for a political community that promotes a narrow, shared cultural and moral identity, often framing these issues in a way that appeals to fear and exclusionary instincts\n \
      Political hate and polarisation (Anti-woke): Narratives that frame efforts to address social inequalities—such as racism, sexism, and anti-LGBTQ+ discrimination—as exaggerated, illegitimate, or a threat to societal norms. These narratives often shift focus away from marginalized groups and instead position privileged groups, such as wealthy individuals, as victims of 'wokeness' or cultural shifts. They emphasize cultural divisions, dismiss systemic inequalities as non-issues, and portray initiatives promoting minority rights or inclusion as oppressive, unnecessary, or harmful\n \
      Religion-related (Anti-Islam): Criticising Islam and portraying it as a threat to national security and cultural identity. Also, claiming that Muslim communities pose a risk to public safety and that the spread of Islam in Europe could lead to the destruction of traditional Western values and cultural heritage\n \
      Religion-related (Anti-Semitic conspiracy theories): Narratives promoting conspiracy theories that allege Jewish people or organisations are responsible for major global events or issues\n \
      Religion-related (Interference with states' affairs): Narratives of foreign religious authorities interfering in a country's internal political processes\n \
      Gender-related (Language-related): Narratives that claim certain gender terminologies or vocabularies are being imposed on individuals or communities. Controversy over language use and inclusivity\n \
      Gender-related (LGBTQ+-related): Narratives related to the LGBTQ+ community. Those include discussions around issues of acceptance, rights, discrimination, etc\n \
      Gender-related (Demographic narratives): Narratives that argue against masturbation or the 'gender propaganda' and claiming that they promote infertility and population decline\n \
      Ethnicity-related (Association to political affiliation): Narratives that associate politicians with specific ethnic groups and label them in negative ways. Elections are viewed through the lens of a competition between different ethnic groups\n \
      Ethnicity-related (Ethnic generalisation): Narratives that generalise entire ethnic groups, treating them as homogeneous 'others'\n \
      Ethnicity-related (Ethnic offensive language): Narratives that use offensive language to describe or address individuals based on their membership in certain ethnic groups\n \
      Ethnicity-related (Threat to population narratives): Narratives claiming that the native population is being replaced by immigrants of different ethnicities\n \
      Migration-related (Migrants societal threat): Narratives that portray migrants as a societal threat, emphasising issues such as violence, terrorism, inequality, and crime\n \
      Distrust in institutions (Failed state): The country is in a state of demise/collapse. Focusing on issues such as immigration, police, etc\n \
      Distrust in institutions (Criticism of national policies): Criticising national policy positions on contentious issues like pension, military conscription, and wealth disparity\n \
      Distrust in democratic system (Elections are rigged): Narratives alleging election manipulation through various means, including political interference, exclusion, anti-vote measures, lack of transparency, and distrust in electoral institutions or polling agencies\n \
      Distrust in democratic system (Anti-Political system): Narratives critical of the political system, asserting that political parties are neither beneficial nor necessary for a well-functioning country\n \
      Distrust in democratic system (Anti-Media): Narratives critical of the media, including attacks on journalists, allegations of corruption or complicity in undermining the country, lack of balance and impartiality, and claims of favoritism towards a particular political faction\n \
      Distrust in democratic system (Immigrants right to vote): Narratives surrounding immigrants' right to vote that claim their participation biases election outcomes or that they are deliberately excluded from the process, with potential impacts on election fairness and representation\n \
      Geopolitics (Pro-Russia): Narratives that shows Russia as strong country that is the victim of the west\n \
      Geopolitics (Foreign interference): Narratives claiming that foreign actors are meddling in the affairs of EU countries\n \
      Geopolitics (Anti-international institutions): Narratives that oppose or criticise organisations such as NATO\n \
      Anti-Elites (Soros): Narratives targeting George Soros, accusing him of manipulating political processes\n \
      Anti-Elites (World Economic Forum / Great Reset): Narratives claiming that the World Economic forum has/seeks control over society\n \
      Anti-Elites (Antisemitism): Narratives accusing Jews or Jewish organisations of having disproportionate power and influence over society and politics\n \
      Anti-Elites (Green Agenda): Narratives that criticise the 'Green Agenda' and green activism\n \
      None: None of the above narratives\n \
      You should not provide any explanation or justification."+"\n"+question+'\n'+"Tweet: "+tweet+'\n'

   
    output = GPT_call(prompt)
    print(output)
    if output in narratives_list:
       LLM_label.append(output)
       print(output)
       LLM_original_label.append(output)
    else:
       print("new label: ", output)
       LLM_label.append('None')
       LLM_original_label.append(output)

dev_set['LLM_label']=LLM_label
dev_set['LLM_original_label']=LLM_original_label
dev_set.to_csv("results/Test_GPT-4o.tsv",sep="\t")

#evaluate
prediction=dev_set["LLM_label"].tolist()
actual=dev_set["label"].tolist()
accuracy = accuracy_score(actual, prediction)
print("Accuracy", round(accuracy, 3))
F1= f1_score(actual, prediction,average="macro")
print("Macro F1_score", round(F1, 3))
F1= f1_score(actual, prediction,average="micro")
print("Micro F1_score", round(F1, 3))
Rec= recall_score(actual, prediction,average="macro")
print("Macro Recall", round(Rec, 3))
Rec= recall_score(actual, prediction,average="micro")
print("Micro Recall", round(Rec, 3))
Prec= precision_score(actual, prediction,average="macro")
print("Macro Precision", round(Rec, 3))
Prec= precision_score(actual, prediction,average="micro")
print("Micro Precision", round(Rec, 3))
