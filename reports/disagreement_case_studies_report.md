# Case Studies: Attention and Predictions on Real Annotator Disagreement

I picked 6 items where D3CODE's own raters actually disagreed by demographic group, not items I guessed would be controversial. 3 were picked because men and women rated them very differently (gender axis), and 3 because raters from Western Europe and the Arab Culture region rated them very differently (region axis). For each one I ran the baseline prompt, then the two matching demographic framings, and checked whether the prediction changed and where attention went.

## The headline result: the model mostly doesn't move at all

Out of 18 condition-runs (6 items x up to 3 conditions each), only 3 actually flipped the prediction. 15 out of 18 stayed exactly where the baseline already was, even on items where real annotators split hard by demographic group (all 6 items have a human disagreement gap of 0.53-0.77, meaning one group rated it offensive 50-77 percentage points more often than the other). So the first honest finding here is: adding one framing sentence rarely moves this model off a decision it already made confidently, even on exactly the items where humans disagree the most.

## Where it did flip, it didn't track the human disagreement direction

This is the more interesting part.

**Item 1710** ("Why would people be posting about the liberals when Paris just had a tradegy?..."): real men rated this offensive 6.7% of the time, real women rated it offensive 83.3% of the time. That's the biggest human gap in the whole set. At baseline the model says not offensive (0), matching men's judgment more than women's. But both gender_man and gender_woman flip it to offensive (1). Same direction, both framings. If the model were adopting the named group's actual perspective, I'd expect gender_man to stay at 0 (matching real men) and only gender_woman to flip. Instead both move the same way. That looks like a generic "this item is right on the boundary and any demographic sentence pushes it over," not the model tracking who it's supposedly talking to.

**Item 143** ("There's always room for a second center-left party. Just no room for extreme right-wingers."): real Western Europe raters called this offensive 10% of the time, real Arab Culture raters called it offensive 80% of the time. At baseline the model already says offensive (1), matching the Arab Culture raters. Naming Western Europe leaves it at 1. But naming Arab Culture flips it to 0, away from what the real Arab Culture raters said, even though the model already agreed with them before we said anything. This is the one case in this whole case-study set where the framing clearly changes the decision, and it moves in the wrong direction if you'd expect "tell it who it's for" to align the output with that group's real judgment.

Items 1494, 4071, 99, and 548 didn't flip at all. All four had a baseline prediction that already matched the more sensitive group's judgment (offensive, matching whichever group rated it higher), and adding either framing left it there.

## Demographic attention share looks like a fixed property of the phrase, not of the item

Here's the pct_demographic_norm (sink-excluded share of attention on the demographic phrase) for all 6 items:

| item | axis | condition | pct_demographic_norm | human gap direction |
|---|---|---|---|---|
| 1710 | gender | man | 0.0579 | women rated it more offensive |
| 1710 | gender | woman | 0.0626 | |
| 1494 | gender | man | 0.0576 | men rated it more offensive |
| 1494 | gender | woman | 0.0626 | |
| 4071 | gender | man | 0.0538 | men rated it more offensive |
| 4071 | gender | woman | 0.0594 | |
| 99 | region | western_europe | 0.0665 | Arab Culture rated it more offensive |
| 99 | region | arab_culture | 0.0875 | |
| 548 | region | western_europe | 0.0758 | Arab Culture rated it more offensive |
| 548 | region | arab_culture | 0.0954 | |
| 143 | region | western_europe | 0.0780 | Arab Culture rated it more offensive |
| 143 | region | arab_culture | 0.1300 | |

Two things jump out. First, woman gets more attention than man on all 3 gender items, and Arab Culture gets more attention than Western Europe on all 3 region items, regardless of which real group actually found that specific item more offensive. That matches the general asymmetry from the main report exactly, and it holds up here even though these are items chosen specifically because human judgment splits by group. So this asymmetry looks like something about the phrase itself (word frequency, tokenization, whatever it is), not something that adapts to which group's perspective would actually matter for that item.

Second, item 143's arab_culture attention share (0.1300) is the highest of all 18 rows by a clear margin, everything else sits between 0.054 and 0.095. And 143 is also the one item where the framing actually flipped the decision. That's a real pattern worth naming, but it's a single data point, not something to lean on as a finding by itself.

## What I'd actually say for the thesis

- On items with the strongest real annotator disagreement, this model's decision is mostly stable under demographic framing. It changes 3 times out of 18, not zero, but not often either.
- Attention on the demographic phrase does not obviously track which real-world group's judgment the item is closer to. It looks driven by the phrase itself more than by the item.
- The one item that did flip in a directionally interpretable way (143) flipped away from, not toward, alignment with the group's real judgment. That's worth stating plainly: adding demographic framing is not a reliable way to make the model agree with that demographic group's actual annotators, at least not in this small sample.
- The other flip (1710) looks like a boundary-sensitivity effect rather than a group-specific one, since both genders push the same direction. That's a real limitation to flag: with only baseline vs. one demographic framing, you can't tell "this changed because of the specific demographic" from "this changed because I added any extra sentence at all." A clean way to check that would be a neutral control condition, a filler sentence with no demographic content ("You are a person.", or something similarly empty), run on the same borderline items, to see if it flips them too. I haven't built that yet. It's cheap to add if you want it before presenting this.

## One honest caveat on sample size

Six items is enough to say "here's what happened in these specific cases," not enough to say "this is how the model behaves on high-disagreement items in general." Everything above should be framed as case studies illustrating a question, not as a statistically powered claim the way the 30-item / 29-item results in the main report are. If you want a general claim here, the next step would be running this same pipeline on a larger, disagreement-selected sample instead of just 3 per axis.
