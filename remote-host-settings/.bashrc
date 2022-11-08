# stop terminal from hanging when saving ctrl-s 
stty -ixon

# use vim bindings
set -o vi
bind '"jj":vi-movement-mode'

# load github keys
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github

# ctrl-l to clear the terminal
bind -x '"\C-l": clear;'

# Git aliases
alias gcm='git commit -m'
alias gpl='git pull'
alias glc='git add .; git commit --amend --no-edit' # add changes to last commit (glc git last commit)
alias glg='git log --stat' 
alias gco='git checkout' 
alias gb='git branch' 
alias gst='git status' 

