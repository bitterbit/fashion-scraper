case "$(uname -s)" in
	   Darwin) # Mac OS :(
		   pwdcmd="PWD";;
	   Linux) # Linux :)
		   pwdcmd="pwd";;
	   *)
		   pwdcmd="pwd";;
esac

curdir=$(eval $pwdcmd)
docker run --rm -v $curdir/src:/src -it fashion-scraper /bin/bash
