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

st.write("I thought I'd try to give it a try with Markov chains and Morgan fingerprints. \n\n" \
"Markov chains are a very straightforward way of modelling evolving sequences. At its"
" simplest, markov chains boil down to just keeping a tally of what is the most likely next thing after the current thing." \
" For text, it's simple enough " \
"to keep a running tally of which words follow which words and bam! You have a lookup table to implement a Markov chain.\n\n" \
"On the otherside, Morgan fingerprints are bitwise vector (lists of 0s and 1s) descriptors of molecules where each bit corresponds to whether or not a functional group is present within a molecule. " \
"Morgan fingerprints are kind of the workhorse of chemoinformatics and that's because they're really flexible." \
" Sometimes you're interested in 2-atom functional groups like alcohol and sometimes you might be interested in 4-atom functional groups like aldehyde. You " \
"can tell the algorithm that generates the fingerprints exactly what size of functional group you're interested in and how many bits you'd like in the descriptoin and gets to work.\n\n"
"So using Morgan fingerprints, I first break up a dataset of molecules into their component functional groups, " \
"then I do a 1:1 mapping of functional groups to words based off of popularity. The most popular functional group gets assigned to the most popular word, the second most to the second most, etc. " \
"While a bit clumsy, it does capture relative importance."
"So very simplly ")


# st.subheader("Markov chains and Morgan fingerprints?")
# st.write("Markov chains and Morgan fingerprints. " \
# "Markov chains")

st.header("2. Molecular Walk and Word Selection")
st.latex(r"P_{\text{final}} = (1 - a) P_{\text{markov}}(w_{i}|w_{i-1}) + a P_{\text{mol}}(w_{i}|m_{i})")

st.header("Where to next?")
st.write("At its base it's")