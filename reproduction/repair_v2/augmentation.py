"""Authored training/validation families; not real-bank benchmark messages."""
import pandas as pd
BANKS=['Access Bank','GTBank','Zenith Bank','UBA','First Bank','FCMB','Kuda','OPay','PalmPay','Moniepoint','Fidelity Bank','your bank','our bank','']
TRAIN_NEG=[
('Funds received','A sum of NGN {amount} has arrived in account {account}. {bank} sent this automated notice for your records. You do not have to respond.'),
('Purchase notification','You spent NGN {amount} with your debit card at Sample Store. The transaction was completed. See your transaction history in the {bank} app.'),
('Account activity','An incoming transfer of NGN {amount} was recorded for account {account}. Reference SAMPLE{account}. This is an informational update from {bank}.'),
('Money sent','You paid NGN {amount} to a beneficiary. The payment has been completed by {bank}. Keep this receipt for reference.'),
('Balance information','Account {account} shows an available balance of NGN {amount}. Review recent activity by opening your usual {bank} mobile application.'),
('Statement available','Your {bank} statement for the previous period is ready to view. Launch your banking application to download it. We do not require an email response.'),
('Service interruption','Some {bank} services will be unavailable for routine system work this weekend. Normal service will resume on Sunday. Thank you for understanding.'),
('Login information','A new sign-in was recorded for your {bank} account. If this was not you, contact the number on your bank card. Do not reply with personal security details.'),
('Access details changed','Your {bank} login password has been reset. No further steps are necessary if you authorised the change. Use the support option in your app for help.'),
('Customer records','To complete a KYC review, bring your identity document to a {bank} branch. Security codes and card PINs must not be sent by email.'),
('Keep your account private','Do not give your card number, CVV, PIN or online banking password to anyone who contacts you. {bank} will not request them through email.'),
('Security learning','A scam message could instruct you to send a password or disclose an OTP. Treat that instruction as dangerous and report it. This is advice from {bank}.'),
('Privacy reminder','If a stranger tells you to provide your PIN for a refund, refuse. Never respond with a password or a token. {bank} encourages you to report such messages.'),
('Requested login code','The code for your requested {bank} sign-in is {account}. Keep the code private and do not forward it. Ignore this notice if you did not request a code.'),
('Code notification','Use {account} to authorise the action you initiated in your {bank} app. This code must never be disclosed to anyone.'),
('Successful transfer','NGN {amount} was transferred successfully from your {bank} account. Transaction reference TEST{account}. This message is for information only.'),
('Credit confirmation','Your {bank} account has been credited with NGN {amount}. Sender: Example Company. Date: 23 October. There is nothing you need to submit.'),
('Debit confirmation','Your {bank} account was debited by NGN {amount}. Narration: card purchase. Check your usual app for further details; no reply is needed.'),
('Project update','The next project review will take place on Tuesday. Please prepare your progress report and bring your questions to the team discussion.'),
('Acknowledgement','Thank you for your enquiry to {bank}. We have received your request and will respond through the usual support channel. Do not send credentials in a reply.'),
('Awareness notice','Never enter your OTP into a link sent by an unknown person. A genuine {bank} employee will never ask you to share your PIN or password.'),
('Anti-scam example','Fraudsters sometimes say: reply with your password now. This is an example of an instruction to avoid, not an instruction from {bank}.'),
('Transfer update','We have completed your transfer request of NGN {amount}. Your {bank} account activity now reflects the payment. No additional verification is required.'),
('Account advice','For your protection, open the banking app yourself when checking payments. Do not follow unsolicited links or send your security codes to callers.')]
TRAIN_POS=[
('Required account review','To finish the {bank} account review, send your online banking password and the token code to our service agent.'),
('Card reactivation','Please reply with your complete debit card number, expiration date and CVV so we can reactivate your {bank} card.'),
('Card replacement','Kindly provide the card PIN and security code to our {bank} representative. We need them to issue your replacement card.'),
('Customer update','Respond to this message with the full sixteen digit card number and your card PIN. Our {bank} update team will complete the changes.'),
('Verification process','Enter your username and password on https://profile-update.invalid/review to complete the {bank} verification process.'),
('Account assistance','Our {bank} assistant needs the OTP from your phone. Share the one-time code with us in your reply.'),
('Support request','Abeg help us send the token wey enter your phone so we fit unlock your {bank} account.'),
('Payment release','Your payment is waiting. Supply your login password and OTP to release NGN {amount} from {bank}.'),
('Refund processing','For a refund of NGN {amount}, tell us the CVV printed on your card and the full card number. This is required by our {bank} desk.'),
('Online identity check','Use https://bank-identity.invalid to enter your {bank} password and SMS code. Your profile cannot remain active without this check.'),
('Customer support','Please send me your password. I am the {bank} representative handling your support ticket.'),
('Account recovery','Disclose the four digit PIN for your {bank} card in a response to this email so we can recover your account.'),
('Priority verification','We will never ask by phone. For this {bank} email verification, reply with your password and verification code.'),
('Protect your money','Never share a code with strangers. Our {bank} team is authorised: submit your OTP and PIN to us now.'),
('Security team','This is {bank} security. Provide your online banking username, password and current token for investigation.'),
('Service follow-up','We appreciate your patience. To complete the {bank} service request, respond with your card number, CVV and PIN.'),
('Update your profile','Log in using your {bank} credentials at https://customer-documents.invalid/account and then type the OTP from your phone.'),
('Telephone confirmation','Send the security code you received to this email address. The {bank} operator is waiting to confirm your ownership.'),
('Urgent verification','Your {bank} account is restricted. Verify your password and PIN at https://access-service.invalid/validate before midnight.'),
('BVN completion','Your BVN validation needs attention. Share your account password and the SMS verification code so {bank} can finish the linkage.'),
('Polite request','Would you please provide your card number and CVV for our {bank} review? We will handle the information carefully.'),
('Token needed','We need your {bank} authentication token. Copy the digits from your phone and email them to our account officer.'),
('Account authorisation','To prevent a restriction, input your {bank} password at https://security-review.invalid then submit your card PIN.'),
('Payment cancellation','You can cancel this unexpected payment by sending your {bank} login password and token to our support mailbox.')]
VAL_NEG=[
('Payment record','A payment of NGN {amount} has been logged against account {account}. Retain this automated {bank} notification. There are no further requirements.'),
('New deposit','NGN {amount} is now in your {bank} account following a completed deposit. You can see the details in your normal account history.'),
('Card spending notice','A merchant charged NGN {amount} to your {bank} debit card. This email is a receipt for the completed purchase.'),
('Annual statement','The annual summary of activity is ready within your {bank} dashboard. Sign in through the application you normally use to read it.'),
('Branch verification','Bring proof of address to your nearest {bank} branch for the account records review. Do not disclose your private login codes by email.'),
('Scheduled upgrade','{bank} will perform a planned technology upgrade on Saturday evening. Payments may be delayed temporarily. No customer response is requested.'),
('Code safety','You requested a transaction in the {bank} app. Your code is {account}. Keep it secret. Staff do not need you to email the code to them.'),
('Avoid disclosure','Do not reply with a debit card number or CVV. Anyone claiming that {bank} requires those details by email is misleading you.'),
('Example of fraud','A fraudster may tell you to submit a password and an SMS code for account recovery. Do not obey that request. {bank} is sharing this warning for your awareness.'),
('Research schedule','The research presentation is scheduled for next week. Our group will meet beforehand to review the methods and answer questions.')]
VAL_POS=[
('Resolve the issue','Give our {bank} agent your complete card details, including CVV and PIN, to resolve this account problem.'),
('Refund confirmation','Reply to confirm the digits on your debit card, the expiry date and security code. This lets our {bank} team send your refund.'),
('Account form','The {bank} recovery form at https://profile-help.invalid needs your password and the code displayed by your token device.'),
('Code verification','For this {bank} customer check, forward the OTP text you received to our support address.'),
('Balance recovery','To recover your {bank} funds, kindly disclose your online account password to the officer assigned to you.'),
('Reactivation','Return the SMS code to us in an email response. We must have that code to reactivate your {bank} profile.'),
('Card assistance','Supply us with your card security code and full card number. A {bank} employee will arrange the card upgrade.'),
('Trusted support','For security never share passwords with callers. Our official {bank} email desk needs you to send your login password and OTP here.'),
('Quick support','Oga send us your {bank} PIN and token code make we complete the verification today.'),
('Customer follow up','Visit https://bank-followup.invalid to confirm your {bank} login password. Type the OTP you receive on the next page.')]
def examples(which):
 neg,pos=(TRAIN_NEG,TRAIN_POS) if which=='train' else (VAL_NEG,VAL_POS);rows=[]
 for label,ts in [(0,neg),(1,pos)]:
  for i,(s,b) in enumerate(ts):
   for k,bank in enumerate(BANKS):
    for greeting in ['', 'Dear customer, ']:rows.append(dict(subject=s if k%3 else '',body=greeting+b.format(bank=bank,amount=[12500,7830,40000][k%3],account=['104829','570326','392801'][k%3]),label=label,family=f'{which}_{label}_{i}',origin='authored_augmentation'))
 return pd.DataFrame(rows)
