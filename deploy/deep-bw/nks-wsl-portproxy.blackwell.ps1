$listen="10.42.0.7"; $ports=@(5562)
$ip=((wsl -e bash -c "hostname -I") | Out-String).Trim().Split(" ")[0]
if (-not $ip) { exit 1 }
foreach ($p in $ports) {
  $cur=(netsh interface portproxy show v4tov4 | Select-String "^\s*$listen\s+$p\s+(\S+)\s+$p").Matches
  if ($cur.Count -and $cur[0].Groups[1].Value -eq $ip) { continue }
  netsh interface portproxy delete v4tov4 listenaddress=$listen listenport=$p 2>$null | Out-Null
  netsh interface portproxy add v4tov4 listenaddress=$listen listenport=$p connectaddress=$ip connectport=$p | Out-Null
}
