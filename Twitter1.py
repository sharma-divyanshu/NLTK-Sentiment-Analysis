from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s
import _thread

# consumer key, consumer secret, access token, access secret.
ckey = "fPby7ACo5SzPrjVS5rLGzBMjt"
csecret = "FLpLTYcCaJneDvX6Mekc5R2AyeZRYSD62aTzluEx575PwBTA2k"
atoken = "550763849-sa6Uot030IC5sT1PBwg0X7HxkO1DLf4uQDeUfkEf"
asecret = "uuTvoTWTL24BP1WHcuN7J49aKgg6WPO98aUUgxPlX43Su"


class listener(StreamListener):

    def on_data(self, data):
        def tweet2json(data):
            tweet_data = json.loads(data)
        try:
            _thread.start_new_thread(tweet2json(data),data)
            tweet = ascii(tweet_data["text"])
            sentiment_value, confidence = s.sentiment(tweet)
            print(tweet, "\n\n", sentiment_value, confidence)
            if confidence*100 >= 80:
                output = open("twitter-out.txt", "a")
                output.write(sentiment_value)
                output.write('\n')
                output.close()

            return True

        except:
            return True



    def on_error(self, status):
        print(status)


auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["good"])
