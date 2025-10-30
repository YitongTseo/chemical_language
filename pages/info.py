# pages/1_White_Paper.py
import streamlit as st

st.title("Chemical Language")
st.header("Why?")
st.write("There are a lot of things on our planet which don't have a mouth or ears or eyes. " \
"I was thinking about how one might construct some sort of universal language that might be interpretible to anything" \
" and I thought maybe a language of chemistry could do it! Encoding meaning directly into matter means any being made of matter can interact with it, " \
" interpret it, maybe even understand it. The recipient of a message could be a tree which perceives time at a different speed than humans, or a bacterium without sense organs at the human scale, " \
"or aliens - why not, and even entities we don't generally label as living such as the ocean."\
" Really, the only ones left out of the fun are beings made of anti-matter who will unfortunately have to wait until we first decide whether antimatter exists or not.\n\n" \
"After some research, I couldn't find any constructed languages based on chemistry or even any language based on any sort of physical laws. " \
"So I thought I'd give it a shot. Here's my first stab at a translation tool for a language with the grammar of chemistry. The first constructed language to be dictated by physical laws.")

st.header("How's it Work?")
st.write("The way I've been thinking about it, at its core this translation challenge boils down to a mapping problem. How can we map semantic meaning to chemical" \
" structure while maintaining some sembalence of \"smoothness\" (similar molecules should be assigned similar meanings) & \"consistency\" " \
"(assigned meanings should reflect deeper chemical properties like acidity, reactivity, etc.)"\
" Without these the translation starts to feel rather arbitrary, a garble of random words pasted onto random molecular bits.")

st.image(
    'early_mapping.png',
    caption='Example of "smooth" mapping where similar molecules are assigned similar(ish) meanings',
)

st.subheader("Markov chains and Morgan fingerprints")

st.write("Markov chains are a very straightforward way of modelling evolving sequences. At its"
" simplest, markov chains boil down to just keeping a tally of what is the most likely next thing after the current thing." \
" For text, it's simple enough " \
"to keep a running tally of which words follow which words and bam! You have a lookup table to implement a Markov chain."\
" Once you have a Markov chain, you can quickly generate new sequences by iteratively selecting each subsequent token by looking at the current token and drawing from the probabilities of which tokens are most likely to come after. I hope this poor explanation makes some at least some amount of sense. \n\n" \
"On the otherside, Morgan fingerprints are bitwise vector (lists of 0s and 1s) descriptors of molecules where each bit corresponds to whether or not a functional group is present within a molecule. " \
"Morgan fingerprints are kind of the workhorse of chemoinformatics and that's because they're incredibly flexible." \
" Functional groups are are short chemical motifs within a molecule. Largely, the study of organic chemistry is the study of functional groups;"
" each functional group has distinct physiochemical properties, interacting with one another, and chemists will often design molecules by kind of assembling and functional groups to decorate a molecule to give it specific properties. "\
" In otherwords, functional groups are kind of the building block of molecules (akin to words in a sentence)."\
" Functional groups come in different sizes' sometimes you're interested in 2-atom functional groups like alcohol and sometimes you might be interested in 4-atom functional groups like aldehyde. One can simply " \
"tell the algorithm that generates Morgan fingerprints exactly what size of functional group they're interested in and how many bits they'd like in the description and the algorithm does the rest. There are some additional details around bit hashing & how the algorithm handles graphical message passing but we're going to glaze past that.\n\n")

st.subheader("My approach")
st.markdown("The mapping I've created here takes a dataset of text and a dataset of molecules (in SMILES string format)." \
"First, using Morgan fingerprints, I break up each molecule into 9ts component functional groups and take a tally how often each functional group appears across the entire pool of molecules. " \
"I do the exact same thing for words in the text dataset and then, very simply, create a correspondence table mapping each functional group to a word based purely off ranked commonality, i.e., the most common word gets mapped to the most common molecule, the second most common gets mapped to the second most common, etc. " \
"I'll be the first to admit this is a pretty clumsy approach, but it does capture relative importance and perhaps in doing so captures the \"flavor\" of each dataset - what makes this specific text corpus and this specific moleculear dataset unique among all text and molecule datasets. "\
"This mapping of a given functional group ($m_i$) to a given word ($w_i$) I'm calling the conditional distribution $P_{\text{mol}}(w_{i}|m_{i})$. You might notice, the distribution is rather spikey (one hit per molecule) to that I say please go ahead and get in there to make it less spikey! I think this can probably be done in a number of ways, the simplest might involve creating some distance metric between similar functional groups. "\
"I will additionally note, that I generate the Morgan fingerprints in this project to specifically describe 2-atom functional groups aka *bonds* within the molecule. This was so I can map the words directly onto the bonds for the molecular visualization and I think it's a rather clean approach.\n\n"
"Next step is to create some text corpus based distribution over words so that the generated translation can flow and not just be word soup (which would naturally arise if all you're doing is grabbing individual words that correspond to individual functional groups). For this I use the humble Markov chain to get a conditonal probability of each next word given the current word aka $P_{\text{markov}}(w_{i}|w_{i-1})$. " \
"One thing to note, is that because we're generating words for each bond in a molecule, the semantic structure is nonlinear but rather a graph. Simple enough, I just run the Markov generation process as a breadth first traversal starting from a random first node." \
" The starting word is given by the functional group (i.e., drawn directly from $P_{\text{mol}}(w_{i}|m_{i})$) and from there each next word is drawn according to the Markov lookup. If the generation ever gets stuck in a deadend, I just start from a new random location in the graph." \
" Finally, to balance the molecular based generation and the text based generation, I string together a very simple weighted sum across probabilities: ")
st.latex(r"P_{\text{final}} = (1 - a) P_{\text{markov}}(w_{i}|w_{i-1}) + a P_{\text{mol}}(w_{i}|m_{i})")
st.markdown("This $a$ term is what I call the Gibberish slider in the web app. When $a$ tends towards 0 we draw more from the Markov chain and grammar is better preserved, when $a$ tends to 1 we give more weight to the chemical makeup of the graph and words are based more upon functional group correspondence.")
st.header("Where next?")
st.write("Believe it or not, but this is not the first approach I've tried to create a language with the grammar of chemistry. The first approach was a much heavier neural network based framework." \
" It was an attempt to also create a mapping but this time not unidirectional (from molecule --> text), but bidirectional (molecule --> text & text --> molecule). " \
"The way I tried to pull it off was to create a joint autoencoder architecture where words and molecules are both mapped to a joint latent space according to some loss function. " \
"Using out of the box encoder/decoders such as BERT, GPT-2, Morgan Fingerprints, and Junction-Tree VAEs I tried my best and actually got something that trained reasonably well. However there were many issues with such a heavy approach. " \
"First off it's pretty nontrivial to train on a new bed of words and molecules. Second, it's a black box so I can't extract out which words necessarily map to which functional groups or really have much understanding of how its doing the mapping at all outside of some simple clustering of outputs and latent space visualization." \
" The third, and final somewhat cincher issue was that it was unwieldy and hard to fine tune. Either translated words were too short or the mappings kind of collapsed and there was little I could do to address it besides fiddling with training data. So I went with an ultimately simpler, but more tunable approach (this one). \n\n" \
"I still think having a bidirectional mapping would be very cool, and so will want to work towards that in the future. Also, I think including reaction mechanics and thermodynamics into the translation process would be amazing. Currently reactions can be simulated by just translating the products and reactants separately.")

st.image(
    'clunky_autoencoder.png',
    caption='Earlier clunky autoencoder architecture',
)

st.write("Let me know what you think! If you have any suggestions or ideas please don't hesistate to reach out :^) \n\n - Yitong (email: yitongt [@] mit [dot] edu)")