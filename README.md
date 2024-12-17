<div align="center">
  <img height="200" src="https://i.ibb.co/FXNsZh8/ringconsole.png"  />
</div>

###

<p align="center">Ring Console<br>A Apex Legends Zone Predicting Discord Bot</p>

###

<p align="center">How It Works:<br><br>use the command "!predict" to analyze a screen shot<br>(note only works if two rings are visable on map)<br><br>the bot will crop out the map, analyze the image for the two rings<br><br>then use math to figure out the next ring location<br><br>if the ring falls under less than 50% playable zone it will find a new zone<br><br>you can react with a :arrows_counterclockwise: to regenrate a new zone if that one isnt to liking</p>

###

<div align="center">
  <img height="600" src="https://i.ibb.co/ryXzKD0/example.png"  />
</div>

###

<p align="center">As you can see from this example its not 100% correct or perfect<br>but it is generally correct in predicting the zones.<br><br>further tweaking will be needed as well as adding more masks for unplayable areas for the remaining maps</p>

###

<p align="center">So how is any of this possible?<br><br><br>the way it works in apex is the last ring is determined first! Then the game calculates the prior rings based on that<br><br>this mean as long as two rings are visable on the map we can determine the next ring based on a simple vector apex uses<br><br>if your curious on the math involved i highly recommend checking out <br>https://github.com/ccamfpsApex/ApexLegendsGuide/wiki/Endzone-&-Ring-Prediction<br><br>this write up is what i based the bot off of and without that info i would be truly lost</p>

###